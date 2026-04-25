"""
Incident Commander Environment - Core Implementation.

Theme 3.1 compliant:
  - Real stateful tool interactions
  - Partial observability
  - Multi-round fix loop
  - Anti-shortcut enforcement
  - Causal chain scenarios
  - Dynamic randomised data each episode
"""

from __future__ import annotations

from uuid import uuid4
from typing import Optional, Set

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

# Resilient imports - work as a top-level script and as a packaged module.
try:
    from models import IncidentCommanderAction, IncidentCommanderObservation
    from simulator import Simulator, CORRECT_FIX
    from rewards import compute_total_reward, MAX_STEPS, MIN_INVESTIGATION_STEPS
except (ImportError, ModuleNotFoundError):
    try:
        from incident_commander.models import IncidentCommanderAction, IncidentCommanderObservation
        from incident_commander.simulator import Simulator, CORRECT_FIX
        from incident_commander.rewards import (
            compute_total_reward, MAX_STEPS, MIN_INVESTIGATION_STEPS,
        )
    except (ImportError, ModuleNotFoundError):
        from ..models import IncidentCommanderAction, IncidentCommanderObservation  # type: ignore
        from ..simulator import Simulator, CORRECT_FIX  # type: ignore
        from ..rewards import (  # type: ignore
            compute_total_reward, MAX_STEPS, MIN_INVESTIGATION_STEPS,
        )


# ---------------------------------------------------------------------------
# Internal state
# ---------------------------------------------------------------------------

class IncidentCommanderState(State):
    """Serialisable state extending openenv State."""
    pass


# ---------------------------------------------------------------------------
# Fix outcome simulator
# ---------------------------------------------------------------------------

def _simulate_fix_outcome(action_type, true_root_cause, locked_hypothesis):
    """Returns 'recovered' | 'degraded' | 'worse'."""
    correct_fix = CORRECT_FIX.get(true_root_cause, "")
    if action_type == correct_fix:
        return "recovered"
    elif action_type in {"restart_pod", "scale_up"}:
        return "degraded"
    else:
        return "worse"


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class IncidentCommanderEnvironment(Environment):
    """AI Incident Commander - Theme 3.1 RL environment."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, difficulty: int = 1):
        super().__init__()
        self._sim = Simulator(difficulty=difficulty)
        self._reset_internal()

    def _reset_internal(self):
        self._state = IncidentCommanderState(
            episode_id=str(uuid4()), step_count=0,
        )
        self._ctx = None
        self._actions = []
        self._locked_hypothesis = None
        self._post_fix_status = None
        self._fix_applied = False
        self._resolved = False
        self._rollback_count = 0
        self._escalation_count = 0
        self._read_logs_set = set()
        self._read_metrics_set = set()
        self._last_result = ""

    def _obs(self, reward=0.0, done=False, breakdown=None):
        assert self._ctx is not None
        steps_used = self._state.step_count
        return IncidentCommanderObservation(
            alert_summary=self._ctx.alert_summary(),
            services_overview=self._ctx.services_overview(),
            revealed_logs=dict(self._ctx._revealed_logs),
            revealed_metrics=dict(self._ctx._revealed_metrics),
            deployment_history=(
                self._ctx.scenario._deployment_history
                if self._ctx._deployment_history_revealed else None
            ),
            dependency_graph=(
                self._ctx.scenario._dependency_graph
                if self._ctx._dependency_graph_revealed else None
            ),
            actions_taken=list(self._actions),
            hypothesis_locked=(self._locked_hypothesis is not None),
            locked_hypothesis=self._locked_hypothesis,
            post_fix_status=self._post_fix_status,
            steps_remaining=MAX_STEPS - steps_used,
            last_action_result=self._last_result,
            reward_breakdown=breakdown,
            done=done,
            reward=reward,
        )

    def reset(self, difficulty=1, seed=None, episode_id=None, **kwargs):
        """Start a new episode. Accepts difficulty either positionally or via kwargs."""
        if 'difficulty' in kwargs:
            difficulty = kwargs['difficulty']
        self._sim.difficulty = max(1, min(4, int(difficulty)))
        self._reset_internal()
        if episode_id:
            self._state.episode_id = str(episode_id)
        self._ctx = self._sim.sample()
        self._last_result = "New incident detected. Read logs and metrics to investigate."
        return self._obs(reward=0.0, done=False)

    def step(self, action, timeout_s=None, **kwargs):
        if self._ctx is None:
            self._ctx = self._sim.sample()

        self._state.step_count += 1
        self._actions.append(action.action_type)

        sc = self._ctx.scenario
        reward = 0.0

        # Timeout guard
        if self._state.step_count >= MAX_STEPS and not self._resolved:
            clamped, breakdown = compute_total_reward(
                post_fix_status=self._post_fix_status,
                locked_hypothesis=self._locked_hypothesis,
                true_root_cause=sc.root_cause,
                actions_taken=self._actions,
                rollback_count=self._rollback_count,
                escalation_count=self._escalation_count,
                already_read_logs=self._read_logs_set,
                already_read_metrics=self._read_metrics_set,
                steps_used=self._state.step_count,
            )
            breakdown["true_root_cause"] = sc.root_cause
            breakdown["root_service"] = sc.root_service
            self._last_result = "[TIMEOUT] Incident unresolved after 50 steps."
            return self._obs(reward=clamped, done=True, breakdown=breakdown)

        atype = action.action_type
        target = action.target_service

        if atype == "read_logs":
            lines = self._ctx.reveal_logs(target, action.time_range_minutes or 5)
            self._read_logs_set.add(target)
            if not lines:
                self._last_result = f"No logs available for {target}."
            else:
                self._last_result = f"Logs for {target}:\n" + "\n".join(f"  {l}" for l in lines)

        elif atype == "read_metrics":
            metrics = self._ctx.reveal_metrics(target)
            self._read_metrics_set.add(target)
            self._last_result = f"Metrics for {target}: {metrics}"

        elif atype == "read_deployment_history":
            history = self._ctx.reveal_deployment_history()
            self._last_result = f"Deployment history: {history}"

        elif atype == "read_dependency_graph":
            graph = self._ctx.reveal_dependency_graph()
            self._last_result = f"Dependency graph: {graph}"

        elif atype == "identify_cause":
            if not action.hypothesis:
                self._last_result = "ERROR: identify_cause requires a hypothesis."
                reward -= 0.05
            elif self._locked_hypothesis is not None:
                self._last_result = f"Hypothesis already locked: {self._locked_hypothesis}"
            else:
                self._locked_hypothesis = action.hypothesis
                self._last_result = f"[OK] Hypothesis locked: {action.hypothesis}"

        elif atype in {"restart_pod", "rollback", "scale_up", "hotfix"}:
            investigation_steps = len(self._read_logs_set) + len(self._read_metrics_set)
            if investigation_steps < MIN_INVESTIGATION_STEPS:
                self._last_result = (
                    f"[BLOCKED] Must read at least {MIN_INVESTIGATION_STEPS} log/metric "
                    f"sources before applying a fix. Investigated: {investigation_steps}."
                )
                reward -= 0.1
                return self._obs(reward=reward)

            if self._locked_hypothesis is None:
                self._last_result = (
                    "[BLOCKED] Must call identify_cause first before applying a fix."
                )
                reward -= 0.1
                return self._obs(reward=reward)

            if atype == "rollback":
                self._rollback_count += 1
                if self._rollback_count > 3:
                    self._last_result = "[BLOCKED] Rollback limit exceeded (>3)."
                    reward -= 0.5
                    return self._obs(reward=reward)

            if not self._fix_applied:
                self._fix_applied = True
                outcome = _simulate_fix_outcome(atype, sc.root_cause, self._locked_hypothesis)
                self._post_fix_status = outcome
                self._last_result = (
                    f"Fix '{atype}' applied to {target}. "
                    f"Call monitor_recovery to observe outcome."
                )
            else:
                outcome = _simulate_fix_outcome(atype, sc.root_cause, self._locked_hypothesis)
                self._post_fix_status = outcome
                self._last_result = (
                    f"Follow-up fix '{atype}' applied to {target}. "
                    f"Call monitor_recovery to observe new outcome."
                )

        elif atype == "monitor_recovery":
            if not self._fix_applied:
                self._last_result = "No fix applied yet. Apply a fix first."
            else:
                status = self._post_fix_status
                if status == "recovered":
                    self._last_result = (
                        f"[OK] {sc.root_service} is HEALTHY. "
                        f"Error rate dropped to 0%. Call resolve to close the incident."
                    )
                elif status == "degraded":
                    self._last_result = (
                        f"[WARN] {sc.root_service} is still DEGRADED. "
                        f"Investigate further or try a different fix."
                    )
                else:
                    self._last_result = (
                        f"[ERROR] {sc.root_service} has WORSENED after the fix. "
                        f"Reconsider your hypothesis."
                    )

        elif atype == "escalate":
            self._escalation_count += 1
            if self._escalation_count > 2:
                self._last_result = "[BLOCKED] Escalation limit exceeded."
                reward -= 0.3
                return self._obs(reward=reward)
            self._last_result = (
                f"Escalated with justification: {action.justification or 'none provided'}."
            )

        elif atype == "resolve":
            if self._post_fix_status != "recovered":
                self._last_result = (
                    "[BLOCKED] Cannot resolve - service has not recovered. "
                    "Apply correct fix and confirm via monitor_recovery first."
                )
                reward -= 0.1
                return self._obs(reward=reward)

            self._resolved = True
            clamped, breakdown = compute_total_reward(
                post_fix_status=self._post_fix_status,
                locked_hypothesis=self._locked_hypothesis,
                true_root_cause=sc.root_cause,
                actions_taken=self._actions,
                rollback_count=self._rollback_count,
                escalation_count=self._escalation_count,
                already_read_logs=self._read_logs_set,
                already_read_metrics=self._read_metrics_set,
                steps_used=self._state.step_count,
            )
            breakdown["true_root_cause"] = sc.root_cause
            breakdown["root_service"] = sc.root_service
            self._last_result = (
                f"[RESOLVED] Incident closed. True root cause: {sc.root_cause}. "
                f"Hypothesis: {self._locked_hypothesis}. "
                f"Reward: {breakdown['total_raw']:.1f} (normalised: {clamped:.3f})"
            )
            return self._obs(reward=clamped, done=True, breakdown=breakdown)

        else:
            self._last_result = f"Unknown action type: {atype}"
            reward -= 0.05

        return self._obs(reward=reward)

    def grade(self) -> float:
        if not self._ctx:
            return 0.01
        sc = self._ctx.scenario
        clamped, _ = compute_total_reward(
            post_fix_status=self._post_fix_status,
            locked_hypothesis=self._locked_hypothesis,
            true_root_cause=sc.root_cause,
            actions_taken=self._actions,
            rollback_count=self._rollback_count,
            escalation_count=self._escalation_count,
            already_read_logs=self._read_logs_set,
            already_read_metrics=self._read_metrics_set,
            steps_used=self._state.step_count,
        )
        score = clamped * 0.98 + 0.01
        return max(0.01, min(0.99, score))

    @property
    def state(self) -> State:
        return self._state
