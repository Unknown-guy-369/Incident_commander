"""
Multi-turn rollout machinery for the Incident Commander environment.

This module is the missing piece that turns the env from a one-shot
"submit one action, get a reward of zero" target into a real RL environment
that exercises every reward signal.

Two clients are provided:

    SyncEnvClient — a tiny `requests`-based client that talks HTTP to a
    deployed OpenEnv server (default: the team's HF Space). Used during
    GRPO training because TRL's reward functions run synchronously.

    LocalEnvAdapter — wraps a local `IncidentCommanderEnvironment` instance
    in the same interface so unit tests and offline development don't need
    network.

Two rollout functions:

    rollout_completion(env_client, prompt, completion) — parses ONE action
    out of `completion` and runs ONE step. Used by single-step rewards.

    rollout_episode(env_client, generate_fn, max_steps, ...) — drives a
    full multi-turn episode by alternating model.generate ↔ env.step until
    `done` or `max_steps`. Used by every reward signal that needs the
    episode to terminate.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

try:
    import requests  # noqa: F401
except ImportError:  # pragma: no cover
    requests = None  # type: ignore


DEFAULT_REMOTE_URL = os.environ.get(
    "INCIDENT_COMMANDER_ENV_URL",
    "https://abishek-priyan-369-incident-commander.hf.space",
).rstrip("/")


VALID_ACTIONS = (
    "read_logs",
    "read_metrics",
    "read_deployment_history",
    "read_dependency_graph",
    "identify_cause",
    "restart_pod",
    "rollback",
    "scale_up",
    "hotfix",
    "escalate",
    "monitor_recovery",
    "resolve",
)


VALID_HYPOTHESES = (
    "memory_limit_too_low",
    "bad_deployment",
    "connection_pool_exhausted",
    "traffic_spike",
    "dependency_failure",
    "config_error",
    "redis_down",
    "certificate_expired",
)


# ---------------------------------------------------------------------------
# Tag parsing
# ---------------------------------------------------------------------------

_ACTION_RE = re.compile(r"<action>\s*([^:<\n]+?)\s*(?::\s*([^<\n]*?))?\s*</action>", re.DOTALL)
_HYPOTHESIS_RE = re.compile(r"<hypothesis>\s*([^<\n]+?)\s*</hypothesis>", re.DOTALL)


def parse_action(completion: str) -> "tuple[Optional[str], Optional[str]]":
    """Returns (action_type, target_service) or (None, None)."""
    m = _ACTION_RE.search(completion)
    if not m:
        return None, None
    atype = m.group(1).strip()
    target = m.group(2).strip() if m.group(2) else ""
    return atype, target


def parse_hypothesis(completion: str) -> Optional[str]:
    """Tries `<hypothesis>x</hypothesis>` first, then falls back to scanning
    the completion for any of the 8 valid root cause tokens."""
    m = _HYPOTHESIS_RE.search(completion)
    if m:
        return m.group(1).strip()
    for h in VALID_HYPOTHESES:
        if h in completion:
            return h
    return None


# ---------------------------------------------------------------------------
# SyncEnvClient — minimal sync HTTP client for an OpenEnv server
# ---------------------------------------------------------------------------

class SyncEnvClient:
    """Drop-in replacement for the async openenv EnvClient when running
    inside GRPO reward callbacks (which are synchronous)."""

    def __init__(self, base_url: str = DEFAULT_REMOTE_URL, timeout: float = 30.0):
        if requests is None:
            raise ImportError("`requests` is required for SyncEnvClient; pip install requests")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    # -- context manager ----------------------------------------------------
    def __enter__(self) -> "SyncEnvClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        self.session.close()

    # -- API ----------------------------------------------------------------
    def reset(self, difficulty: Optional[int] = None) -> Dict[str, Any]:
        body: Dict[str, Any] = {}
        if difficulty is not None:
            body["difficulty"] = difficulty
        resp = self.session.post(f"{self.base_url}/reset", json=body, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def step(
        self,
        action_type: str,
        target_service: Optional[str] = None,
        hypothesis: Optional[str] = None,
        justification: Optional[str] = None,
        time_range_minutes: Optional[int] = None,
    ) -> Dict[str, Any]:
        action: Dict[str, Any] = {
            "action_type": action_type,
            "target_service": target_service or "payment-service",
        }
        if hypothesis:
            action["hypothesis"] = hypothesis
        if justification:
            action["justification"] = justification
        if time_range_minutes:
            action["time_range_minutes"] = time_range_minutes
        resp = self.session.post(
            f"{self.base_url}/step", json={"action": action}, timeout=self.timeout
        )
        resp.raise_for_status()
        return resp.json()


# ---------------------------------------------------------------------------
# LocalEnvAdapter — same surface as SyncEnvClient, but in-process
# ---------------------------------------------------------------------------

class LocalEnvAdapter:
    """Wraps an in-process `IncidentCommanderEnvironment` in the SyncEnvClient
    surface so the same rollout code works offline."""

    def __init__(self, env, difficulty: int = 1):
        self.env = env
        self.difficulty = difficulty

    def __enter__(self) -> "LocalEnvAdapter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        pass

    def close(self) -> None:
        pass

    @staticmethod
    def _obs_to_dict(obs) -> Dict[str, Any]:
        if hasattr(obs, "model_dump"):
            return obs.model_dump()
        if hasattr(obs, "dict"):
            return obs.dict()
        return dict(obs)

    def reset(self, difficulty: Optional[int] = None) -> Dict[str, Any]:
        d = difficulty if difficulty is not None else self.difficulty
        obs = self.env.reset(difficulty=d)
        return {"observation": self._obs_to_dict(obs), "reward": 0.0, "done": False}

    def step(
        self,
        action_type: str,
        target_service: Optional[str] = None,
        hypothesis: Optional[str] = None,
        justification: Optional[str] = None,
        time_range_minutes: Optional[int] = None,
    ) -> Dict[str, Any]:
        # local import to avoid a hard dependency at module load
        try:
            from models import IncidentCommanderAction
        except (ImportError, ModuleNotFoundError):
            from incident_commander.models import IncidentCommanderAction  # type: ignore
        action = IncidentCommanderAction(
            action_type=action_type,
            target_service=target_service or "payment-service",
            hypothesis=hypothesis,
            justification=justification,
            time_range_minutes=time_range_minutes or 5,
        )
        obs = self.env.step(action)
        return {
            "observation": self._obs_to_dict(obs),
            "reward": float(getattr(obs, "reward", 0.0)),
            "done": bool(getattr(obs, "done", False)),
        }


# ---------------------------------------------------------------------------
# Episode state — the dict the reward functions consume
# ---------------------------------------------------------------------------

@dataclass
class EpisodeState:
    """Aggregated terminal state of a multi-turn rollout."""

    post_fix_status: Optional[str] = None
    locked_hypothesis: Optional[str] = None
    true_root_cause: str = ""
    actions_taken: List[str] = field(default_factory=list)
    rollback_count: int = 0
    escalation_count: int = 0
    already_read_logs: Set[str] = field(default_factory=set)
    already_read_metrics: Set[str] = field(default_factory=set)
    steps_used: int = 0
    read_root_service: bool = False
    last_reward: float = 0.0
    last_observation: Dict[str, Any] = field(default_factory=dict)
    transcript: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "post_fix_status": self.post_fix_status,
            "locked_hypothesis": self.locked_hypothesis,
            "true_root_cause": self.true_root_cause,
            "actions_taken": list(self.actions_taken),
            "rollback_count": self.rollback_count,
            "escalation_count": self.escalation_count,
            "already_read_logs": set(self.already_read_logs),
            "already_read_metrics": set(self.already_read_metrics),
            "steps_used": self.steps_used,
            "read_root_service": self.read_root_service,
            "last_reward": self.last_reward,
        }


# ---------------------------------------------------------------------------
# Prompt rendering helper
# ---------------------------------------------------------------------------

def render_observation(obs: Dict[str, Any]) -> str:
    """Compact text form of an observation for inclusion in a prompt."""
    parts = []
    parts.append(f"Alert: {obs.get('alert_summary', '')}")
    services = obs.get("services_overview") or []
    if services:
        names = [s.get("name") for s in services]
        parts.append(f"Services: {names}")
    if obs.get("revealed_logs"):
        parts.append("Logs read so far:")
        for svc, lines in obs["revealed_logs"].items():
            parts.append(f"  [{svc}]")
            for ln in lines[:5]:
                parts.append(f"    {ln}")
    if obs.get("revealed_metrics"):
        parts.append(f"Metrics read so far: {list(obs['revealed_metrics'].keys())}")
    if obs.get("deployment_history") is not None:
        parts.append(f"Deployment history: {obs['deployment_history']}")
    if obs.get("dependency_graph") is not None:
        parts.append(f"Dependency graph: {obs['dependency_graph']}")
    if obs.get("hypothesis_locked"):
        parts.append(f"Locked hypothesis: {obs.get('locked_hypothesis')}")
    if obs.get("post_fix_status"):
        parts.append(f"Post-fix status: {obs.get('post_fix_status')}")
    if obs.get("last_action_result"):
        parts.append(f"Last result: {obs['last_action_result']}")
    parts.append(f"Steps remaining: {obs.get('steps_remaining', '?')}")
    return "\n".join(parts)


SYSTEM_PROMPT = """You are an AI Incident Commander.
You investigate microservice outages by using tools and reasoning causally.

Episode flow:
  1. Read logs / metrics / deployment_history / dependency_graph to investigate.
  2. Call identify_cause with a hypothesis (one of:
     memory_limit_too_low, bad_deployment, connection_pool_exhausted,
     traffic_spike, dependency_failure, config_error, redis_down,
     certificate_expired).
  3. Apply the matching fix (restart_pod, rollback, scale_up, or hotfix).
  4. Call monitor_recovery to observe outcome.
  5. Call resolve once the service has recovered.

Output your reasoning in <thought>...</thought> and your single next
action in <action>action_type:target_service</action>. For identify_cause
also include <hypothesis>name</hypothesis>.
"""


# ---------------------------------------------------------------------------
# Single-step rollout (kept for backward compatibility)
# ---------------------------------------------------------------------------

def rollout_completion(env_client, prompt: str, completion: str,
                       difficulty: Optional[int] = 1) -> EpisodeState:
    """Runs ONE step of the env for the action parsed out of `completion`.

    Used as a fallback when the model emits a single-action turn rather than
    a planned trajectory.
    """
    state = EpisodeState()
    atype, target = parse_action(completion)
    if atype is None or atype not in VALID_ACTIONS:
        return state

    try:
        env_client.reset(difficulty=difficulty)
        kwargs: Dict[str, Any] = {}
        if atype == "identify_cause":
            kwargs["hypothesis"] = parse_hypothesis(completion)
        result = env_client.step(action_type=atype, target_service=target or None, **kwargs)
        obs = result.get("observation", {})
        state.actions_taken = [atype]
        state.steps_used = 1
        state.last_reward = float(result.get("reward", 0.0))
        state.last_observation = obs
        state.locked_hypothesis = obs.get("locked_hypothesis")
        state.post_fix_status = obs.get("post_fix_status")
        if atype == "rollback":
            state.rollback_count = 1
        if atype == "escalate":
            state.escalation_count = 1
        if atype == "read_logs" and target:
            state.already_read_logs.add(target)
        if atype == "read_metrics" and target:
            state.already_read_metrics.add(target)
    except Exception:  # noqa: BLE001
        pass
    return state


# ---------------------------------------------------------------------------
# Multi-turn rollout — the heart of the fix
# ---------------------------------------------------------------------------

def rollout_episode(
    env_client,
    generate_fn: Callable[[str], str],
    max_steps: int = 25,
    difficulty: Optional[int] = 1,
    system_prompt: str = SYSTEM_PROMPT,
    on_step: Optional[Callable[[int, str, Dict[str, Any]], None]] = None,
    deterministic_root_cause: Optional[str] = None,
) -> EpisodeState:
    """Run a full multi-turn episode: model.generate ↔ env.step.

    Args:
        env_client:    SyncEnvClient or LocalEnvAdapter.
        generate_fn:   Callable taking a prompt string and returning the model's
                       completion. The implementation decides decoding params.
        max_steps:     Maximum env steps (defaults to 25 to respect Colab budgets;
                       MAX_STEPS in the env is 50).
        difficulty:    Difficulty to request at reset.
        system_prompt: System prompt prepended to every turn.
        on_step:       Optional callback (step_idx, completion, obs_dict).
        deterministic_root_cause: For unit tests — bypass model and use known cause.

    Returns:
        EpisodeState with terminal aggregates suitable for the reward functions.
    """
    state = EpisodeState()
    init = env_client.reset(difficulty=difficulty)
    obs = init.get("observation", {})
    transcript_so_far = system_prompt
    state.last_observation = obs
    # Best-effort recovery of the true root cause for offline reward replay.
    if deterministic_root_cause is not None:
        state.true_root_cause = deterministic_root_cause

    for step_idx in range(max_steps):
        prompt = (
            transcript_so_far
            + "\n\n--- Current state ---\n"
            + render_observation(obs)
            + "\n\nNext action:"
        )
        completion = generate_fn(prompt)
        atype, target = parse_action(completion)

        # Bookkeeping for malformed completions
        if atype is None or atype not in VALID_ACTIONS:
            state.actions_taken.append("__parse_error__")
            transcript_so_far += f"\nAssistant: {completion}\nObservation: (parse error, agent must retry)"
            if on_step:
                on_step(step_idx, completion, obs)
            continue

        kwargs: Dict[str, Any] = {}
        if atype == "identify_cause":
            kwargs["hypothesis"] = parse_hypothesis(completion)
        if atype in ("escalate", "resolve"):
            kwargs["justification"] = "agent decision"

        try:
            result = env_client.step(
                action_type=atype, target_service=target or None, **kwargs
            )
        except Exception as e:  # noqa: BLE001
            transcript_so_far += f"\nAssistant: {completion}\nObservation: (env error: {e})"
            state.actions_taken.append("__env_error__")
            continue

        obs = result.get("observation", {})
        state.last_observation = obs
        state.last_reward = float(result.get("reward", 0.0))
        state.actions_taken.append(atype)
        state.steps_used += 1

        if atype == "rollback":
            state.rollback_count += 1
        if atype == "escalate":
            state.escalation_count += 1
        if atype == "read_logs" and target:
            state.already_read_logs.add(target)
        if atype == "read_metrics" and target:
            state.already_read_metrics.add(target)
        if atype == "identify_cause":
            state.locked_hypothesis = obs.get("locked_hypothesis") or kwargs.get("hypothesis")

        state.post_fix_status = obs.get("post_fix_status")

        # Lazy guess at root service from the observation, so we can credit
        # `read_root_service` even without a privileged channel.
        if not state.read_root_service:
            services = [s.get("name") for s in (obs.get("services_overview") or [])]
            for svc in (state.already_read_logs | state.already_read_metrics):
                # Approximation: the "root" is the most-degraded service in the overview.
                degraded = [s.get("name") for s in (obs.get("services_overview") or []) if s.get("status") == "degraded"]
                if svc in degraded:
                    state.read_root_service = True
                    break

        transcript_so_far += (
            f"\nAssistant: {completion}\nObservation:\n{render_observation(obs)}"
        )

        if on_step:
            on_step(step_idx, completion, obs)

        if result.get("done"):
            break

    # Final true-root-cause recovery from terminal breakdown if available.
    # The env appends `true_root_cause` and `root_service` to the breakdown
    # only on done=True, so this is safe — the agent never sees them mid-episode.
    bd = (state.last_observation or {}).get("reward_breakdown") or {}
    if not state.true_root_cause and isinstance(bd, dict):
        state.true_root_cause = bd.get("true_root_cause", "") or state.true_root_cause
        root_svc = bd.get("root_service")
        if root_svc and (
            root_svc in state.already_read_logs
            or root_svc in state.already_read_metrics
        ):
            state.read_root_service = True

    return state


# ---------------------------------------------------------------------------
# Convenience wrapper — used by training/notebook
# ---------------------------------------------------------------------------

def make_episode_rollout_fn(
    env_url: Optional[str] = None,
    generate_fn: Optional[Callable[[str], str]] = None,
    max_steps: int = 15,
    difficulty: int = 1,
    use_local: bool = False,
):
    """Returns a `rollout_fn(prompt, completion) -> dict` for use with
    `rewards.make_grpo_reward_fns(...)`.

    The returned callable IGNORES `prompt` (the env handles its own state)
    and IGNORES `completion` (a real episode is generated via `generate_fn`).
    This is the canonical way to drive multi-turn GRPO with the four
    reward signals.
    """
    if generate_fn is None:
        raise ValueError("generate_fn is required — pass a fn that calls model.generate")

    def _factory():
        if use_local:
            try:
                from server.incident_commander_environment import IncidentCommanderEnvironment
            except (ImportError, ModuleNotFoundError):
                from incident_commander.server.incident_commander_environment import (
                    IncidentCommanderEnvironment,
                )
            return LocalEnvAdapter(IncidentCommanderEnvironment(difficulty=difficulty), difficulty=difficulty)
        return SyncEnvClient(env_url or DEFAULT_REMOTE_URL)

    def rollout_fn(prompt: str, completion: str) -> Dict[str, Any]:
        with _factory() as env:
            state = rollout_episode(
                env, generate_fn,
                max_steps=max_steps, difficulty=difficulty,
            )
        return state.to_dict()

    return rollout_fn
