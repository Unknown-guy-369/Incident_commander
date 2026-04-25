"""
Multi-turn rollout machinery for the Incident Commander environment.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

try:
    import requests
except ImportError:
    requests = None  # type: ignore


DEFAULT_REMOTE_URL = os.environ.get(
    "INCIDENT_COMMANDER_ENV_URL",
    "https://abishek-priyan-369-incident-commander.hf.space",
).rstrip("/")


VALID_ACTIONS = (
    "read_logs", "read_metrics",
    "read_deployment_history", "read_dependency_graph",
    "identify_cause",
    "restart_pod", "rollback", "scale_up", "hotfix",
    "escalate", "monitor_recovery", "resolve",
)

VALID_HYPOTHESES = (
    "memory_limit_too_low", "bad_deployment", "connection_pool_exhausted",
    "traffic_spike", "dependency_failure", "config_error",
    "redis_down", "certificate_expired",
)


# ---------------------------------------------------------------------------
# Tag parsing
# ---------------------------------------------------------------------------

_ACTION_RE = re.compile(r"<action>\s*([^:<\n]+?)\s*(?::\s*([^<\n]*?))?\s*</action>", re.DOTALL)
_HYPOTHESIS_RE = re.compile(r"<hypothesis>\s*([^<\n]+?)\s*</hypothesis>", re.DOTALL)


def parse_action(completion):
    m = _ACTION_RE.search(completion)
    if not m:
        return None, None
    atype = m.group(1).strip()
    target = m.group(2).strip() if m.group(2) else ""
    return atype, target


def parse_hypothesis(completion):
    m = _HYPOTHESIS_RE.search(completion)
    if m:
        return m.group(1).strip()
    for h in VALID_HYPOTHESES:
        if h in completion:
            return h
    return None


# ---------------------------------------------------------------------------
# SyncEnvClient
# ---------------------------------------------------------------------------

class SyncEnvClient:
    """Sync HTTP client for OpenEnv server (used inside GRPO reward callbacks)."""

    def __init__(self, base_url=DEFAULT_REMOTE_URL, timeout=30.0):
        if requests is None:
            raise ImportError("requests is required: pip install requests")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def close(self):
        self.session.close()

    def reset(self, difficulty=None):
        body = {}
        if difficulty is not None:
            body["difficulty"] = difficulty
        resp = self.session.post(f"{self.base_url}/reset", json=body, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def step(self, action_type, target_service=None, hypothesis=None,
             justification=None, time_range_minutes=None):
        action = {
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
# LocalEnvAdapter
# ---------------------------------------------------------------------------

class LocalEnvAdapter:
    """Wraps an in-process IncidentCommanderEnvironment with the SyncEnvClient surface."""

    def __init__(self, env, difficulty=1):
        self.env = env
        self.difficulty = difficulty

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass

    def close(self):
        pass

    @staticmethod
    def _obs_to_dict(obs):
        if hasattr(obs, "model_dump"):
            return obs.model_dump()
        if hasattr(obs, "dict"):
            return obs.dict()
        return dict(obs)

    def reset(self, difficulty=None):
        d = difficulty if difficulty is not None else self.difficulty
        obs = self.env.reset(difficulty=d)
        return {"observation": self._obs_to_dict(obs), "reward": 0.0, "done": False}

    def step(self, action_type, target_service=None, hypothesis=None,
             justification=None, time_range_minutes=None):
        try:
            from models import IncidentCommanderAction
        except (ImportError, ModuleNotFoundError):
            from incident_commander.models import IncidentCommanderAction
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
# Episode state
# ---------------------------------------------------------------------------

@dataclass
class EpisodeState:
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

    def to_dict(self):
        bd = (self.last_observation or {}).get("reward_breakdown") or {}
        if not isinstance(bd, dict):
            bd = {}
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
            "reward_breakdown": dict(bd),
        }


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------

def render_observation(obs):
    parts = [f"Alert: {obs.get('alert_summary', '')}"]
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


# ---------------------------------------------------------------------------
# SYSTEM_PROMPT with one-shot worked example
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an AI Incident Commander.
Investigate microservice outages by using tools and reasoning causally.

Output every turn as: a <thought>...</thought> block, then exactly one
<action>action_type:target_service</action> tag.
For identify_cause, also include <hypothesis>name</hypothesis> just before
the <action> tag.

Episode flow:
  1. read_logs and read_metrics on at least 3 different services.
  2. identify_cause with a hypothesis (one of: memory_limit_too_low,
     bad_deployment, connection_pool_exhausted, traffic_spike,
     dependency_failure, config_error, redis_down, certificate_expired).
  3. Apply the matching fix:
     memory_limit_too_low->scale_up, bad_deployment->rollback,
     connection_pool_exhausted->hotfix, traffic_spike->scale_up,
     dependency_failure->restart_pod, config_error->hotfix,
     redis_down->restart_pod, certificate_expired->hotfix.
  4. monitor_recovery to confirm.
  5. resolve to close.

=== Worked example (follow this format exactly) ===
Observation: Alert: payment-service error rate critical
Services: ['api-gateway', 'payment-service', 'auth-service']
<thought>Investigating payment-service first.</thought><action>read_logs:payment-service</action>

Observation: FATAL OOMKilled, memory at 98%.
<thought>Memory issue; check another service.</thought><action>read_logs:auth-service</action>

Observation: auth-service all systems normal.
<thought>Read api-gateway too.</thought><action>read_logs:api-gateway</action>

Observation: api-gateway normal.
<thought>Read metrics on payment-service.</thought><action>read_metrics:payment-service</action>

Observation: cpu=85, memory=98, oom_kills=5.
<thought>Memory limit too low. Lock hypothesis.</thought><hypothesis>memory_limit_too_low</hypothesis><action>identify_cause:payment-service</action>

Observation: Hypothesis locked: memory_limit_too_low.
<thought>The fix for memory_limit_too_low is scale_up.</thought><action>scale_up:payment-service</action>

Observation: Fix applied. Call monitor_recovery.
<thought>Check status.</thought><action>monitor_recovery:</action>

Observation: payment-service is HEALTHY. Call resolve.
<thought>Recovered, closing.</thought><action>resolve:</action>
=== End example ===

Now do the real episode below. One <thought> + one <action> per turn.
"""


# ---------------------------------------------------------------------------
# Single-step rollout (kept for compatibility)
# ---------------------------------------------------------------------------

def rollout_completion(env_client, prompt, completion, difficulty=1):
    state = EpisodeState()
    atype, target = parse_action(completion)
    if atype is None or atype not in VALID_ACTIONS:
        return state
    try:
        env_client.reset(difficulty=difficulty)
        kwargs = {}
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
    except Exception:
        pass
    return state


# ---------------------------------------------------------------------------
# Multi-turn rollout
# ---------------------------------------------------------------------------

def rollout_episode(env_client, generate_fn, max_steps=12, difficulty=1,
                    system_prompt=None, on_step=None,
                    deterministic_root_cause=None):
    """Run a full multi-turn episode: model.generate <-> env.step."""
    if system_prompt is None:
        system_prompt = SYSTEM_PROMPT

    state = EpisodeState()
    init = env_client.reset(difficulty=difficulty)
    obs = init.get("observation", {})
    transcript_so_far = system_prompt
    state.last_observation = obs
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

        if atype is None or atype not in VALID_ACTIONS:
            state.actions_taken.append("__parse_error__")
            transcript_so_far += f"\nAssistant: {completion}\nObservation: (parse error)"
            if on_step:
                on_step(step_idx, completion, obs)
            continue

        kwargs = {}
        if atype == "identify_cause":
            kwargs["hypothesis"] = parse_hypothesis(completion)
        if atype in ("escalate", "resolve"):
            kwargs["justification"] = "agent decision"

        try:
            result = env_client.step(
                action_type=atype, target_service=target or None, **kwargs
            )
        except Exception as e:
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

        if not state.read_root_service:
            degraded = [s.get("name") for s in (obs.get("services_overview") or [])
                        if s.get("status") == "degraded"]
            for svc in (state.already_read_logs | state.already_read_metrics):
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
# Convenience factory
# ---------------------------------------------------------------------------

def make_episode_rollout_fn(env_url=None, generate_fn=None, max_steps=12,
                            difficulty=1, use_local=False):
    if generate_fn is None:
        raise ValueError("generate_fn is required")

    def _factory():
        if use_local:
            try:
                from server.incident_commander_environment import IncidentCommanderEnvironment
            except (ImportError, ModuleNotFoundError):
                from incident_commander.server.incident_commander_environment import (
                    IncidentCommanderEnvironment,
                )
            return LocalEnvAdapter(
                IncidentCommanderEnvironment(difficulty=difficulty),
                difficulty=difficulty,
            )
        return SyncEnvClient(env_url or DEFAULT_REMOTE_URL)

    def rollout_fn(prompt, completion):
        with _factory() as env:
            state = rollout_episode(
                env, generate_fn,
                max_steps=max_steps, difficulty=difficulty,
            )
        return state.to_dict()

    return rollout_fn
