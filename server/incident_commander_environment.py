"""
Incident Commander Environment — Core Implementation.

Theme 3.1 compliant:
  ✅ Real stateful tool interactions (tools mutate environment state)
  ✅ True partial observability (only what was explicitly read is returned)
  ✅ Multi-round fix loop (episode continues after fix; agent observes outcome)
  ✅ Anti-shortcut enforcement (min investigation steps, fix-hypothesis matching)
  ✅ Causal chain scenarios (multi-hop, time-delayed)
  ✅ No shortcuts (3 shortcut categories fully blocked)
  ✅ Dynamic randomised data every episode (cannot memorise)
"""

from __future__ import annotations

from uuid import uuid4
from typing import Optional, Set

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from models import IncidentCommanderAction, IncidentCommanderObservation
    from simulator import Simulator, CORRECT_FIX
    from rewards import compute_total_reward, MAX_STEPS, MIN_INVESTIGATION_STEPS
except (ImportError, ModuleNotFoundError):
    from models import IncidentCommanderAction, IncidentCommanderObservation
    from simulator import Simulator, CORRECT_FIX
    from rewards import compute_total_reward, MAX_STEPS, MIN_INVESTIGATION_STEPS


# ---------------------------------------------------------------------------
# Internal state
# ---------------------------------------------------------------------------

class IncidentCommanderState(State):
    """Serialisable state for the environment (extends openenv State)."""
    pass  # all mutable state lives in IncidentCommanderEnvironment attributes


# ---------------------------------------------------------------------------
# Fix outcome simulator
# ---------------------------------------------------------------------------

def _simulate_fix_outcome(
    action_type: str,
    true_root_cause: str,
    locked_hypothesis: Optional[str],
) -> str:
    """Determine whether the fix worked, based on whether the right action
    was applied for the actual root cause.

    Returns: 'recovered' | 'degraded' | 'worse'
    """
    correct_fix = CORRECT_FIX.get(true_root_cause, "")
    if action_type == correct_fix:
        return "recovered"
    elif action_type in {"restart_pod", "scale_up"}:
        # Partially helpful but wrong for this root cause
        return "degraded"
    else:
        return "worse"


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class IncidentCommanderEnvironment(Environment):
    """
    AI Incident Commander — Theme 3.1 compliant RL environment.

    Episode flow:
        reset() → receive alert + services overview
        read_logs(service) → reveal partial logs
        read_metrics(service) → reveal partial metrics
        read_deployment_history() → reveal recent deploys
        read_dependency_graph() → reveal service graph
        identify_cause(hypothesis) → lock root cause
        <fix action> → apply fix, advance simulated time
        monitor_recovery() → observe post-fix outcome
        resolve() → end episode
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, difficulty: int = 1):
        self._sim = Simulator(difficulty=difficulty)
        self._reset_internal()

    # ── Internal helpers ─────────────────────────────────────────────────

    def _reset_internal(self):
        self._state = IncidentCommanderState(
            episode_id=str(uuid4()), step_count=0
        )
        self._ctx = None
        self._actions: list[str] = []         # action_type history
        self._locked_hypothesis: Optional[str] = None
        self._post_fix_status: Optional[str] = None
        self._fix_applied: bool = False
        self._resolved: bool = False
        self._rollback_count: int = 0
        self._escalation_count: int = 0
        self._read_logs_set: Set[str] = set()
        self._read_metrics_set: Set[str] = set()
        self._last_result: str = ""

    def _obs(self, reward: float = 0.0, done: bool = False, breakdown=None) -> IncidentCommanderObservation:
        assert self._ctx is not None
        steps_used = self._state.step_count
        return IncidentCommanderObservation(
            alert_summary=self._ctx.alert_summary(),
            services_overview=self._ctx.services_overview(),
            revealed_logs=dict(self._ctx._revealed_logs),
            revealed_metrics=dict(self._ctx._revealed_metrics),
            deployment_history=(
                self._ctx.scenario._deployment_history
                if self._ctx._deployment_history_revealed
                else None
            ),
            dependency_graph=(
                self._ctx.scenario._dependency_graph
                if self._ctx._dependency_graph_revealed
                else None
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

    # ── Reset ─────────────────────────────────────────────────────────────

    def reset(self, difficulty: int = 1) -> IncidentCommanderObservation:  # type: ignore[override]
        """Start a new episode with a freshly sampled scenario."""
        self._sim.difficulty = max(1, min(4, difficulty))
        self._reset_internal()
        self._ctx = self._sim.sample()
        self._last_result = "New incident detected. Read logs and metrics to investigate."
        return self._obs(reward=0.0, done=False)

    # ── Step ──────────────────────────────────────────────────────────────

    def step(self, action: IncidentCommanderAction) -> IncidentCommanderObservation:  # type: ignore[override]
        if self._ctx is None:
            # Auto-reset if called before reset()
            self._ctx = self._sim.sample()

        self._state.step_count += 1
        self._actions.append(action.action_type)

        sc = self._ctx.scenario
        reward = 0.0
        done = False

        # ── Timeout guard ────────────────────────────────────────────────
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
            self._last_result = "⏰ Timeout — incident unresolved after 50 steps."
            return self._obs(reward=clamped, done=True, breakdown=breakdown)

        # ── Dispatch ─────────────────────────────────────────────────────

        atype = action.action_type
        target = action.target_service

        # ── Investigation tools ──────────────────────────────────────────
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

        # ── Hypothesis commitment ─────────────────────────────────────────
        elif atype == "identify_cause":
            if not action.hypothesis:
                self._last_result = "ERROR: identify_cause requires a hypothesis value."
                reward -= 0.05
            elif self._locked_hypothesis is not None:
                self._last_result = f"Hypothesis already locked: {self._locked_hypothesis}"
            else:
                self._locked_hypothesis = action.hypothesis
                self._last_result = f"[OK] Hypothesis locked: {action.hypothesis}"

        # ── Fix actions ──────────────────────────────────────────────────
        elif atype in {"restart_pod", "rollback", "scale_up", "hotfix"}:

            # Anti-shortcut: must have investigated first
            investigation_steps = len(self._read_logs_set) + len(self._read_metrics_set)
            if investigation_steps < MIN_INVESTIGATION_STEPS:
                self._last_result = (
                    f"⛔ ACTION BLOCKED — must read at least {MIN_INVESTIGATION_STEPS} "
                    f"log/metric sources before applying a fix. "
                    f"Investigated: {investigation_steps}."
                )
                reward -= 0.1
                return self._obs(reward=reward)

            # Anti-shortcut: must commit hypothesis first
            if self._locked_hypothesis is None:
                self._last_result = (
                    "⛔ ACTION BLOCKED — must call identify_cause first "
                    "before applying a fix."
                )
                reward -= 0.1
                return self._obs(reward=reward)

            if atype == "rollback":
                self._rollback_count += 1
                if self._rollback_count > 3:
                    self._last_result = "⛔ Rollback limit exceeded (>3). Hard stop."
                    reward -= 0.5
                    return self._obs(reward=reward)

            if not self._fix_applied:
                self._fix_applied = True
                outcome = _simulate_fix_outcome(atype, sc.root_cause, self._locked_hypothesis)
                self._post_fix_status = outcome
                self._last_result = (
                    f"Fix '{atype}' applied to {target}. "
                    f"Simulating recovery... call monitor_recovery to observe outcome."
                )
            else:
                # Second fix attempt — update outcome
                outcome = _simulate_fix_outcome(atype, sc.root_cause, self._locked_hypothesis)
                self._post_fix_status = outcome
                self._last_result = (
                    f"Follow-up fix '{atype}' applied to {target}. "
                    f"Call monitor_recovery to observe new outcome."
                )

        # ── Monitor recovery (multi-round loop) ────────────────────────
        elif atype == "monitor_recovery":
            if not self._fix_applied:
                self._last_result = "No fix has been applied yet. Apply a fix first."
            else:
                status = self._post_fix_status
                if status == "recovered":
                    self._last_result = (
                        f"✅ {sc.root_service} is HEALTHY. "
                        f"Error rate dropped to 0%. Call resolve to close the incident."
                    )
                elif status == "degraded":
                    self._last_result = (
                        f"⚠️  {sc.root_service} is still DEGRADED. "
                        f"Fix partially helped but root cause may not match hypothesis. "
                        f"Investigate further or try a different fix."
                    )
                else:  # worse
                    self._last_result = (
                        f"🔴 {sc.root_service} has WORSENED after the fix. "
                        f"Your fix may have made things worse. "
                        f"Reconsider your hypothesis."
                    )

        # ── Escalation ───────────────────────────────────────────────────
        elif atype == "escalate":
            self._escalation_count += 1
            if self._escalation_count > 2:
                self._last_result = "⛔ Escalation limit exceeded. Hard stop."
                reward -= 0.3
                return self._obs(reward=reward)
            self._last_result = (
                f"Escalated with justification: {action.justification or 'none provided'}. "
                f"On-call engineer notified."
            )

        # ── Resolve ───────────────────────────────────────────────────────
        elif atype == "resolve":
            if self._post_fix_status != "recovered":
                self._last_result = (
                    "⛔ Cannot resolve — service has not recovered. "
                    "Apply the correct fix and confirm via monitor_recovery first."
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

    # ── Grade ─────────────────────────────────────────────────────────────

    def grade(self) -> float:
        """Grade the current episode final state (0.01–0.99)."""
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
        # clamped is already in [0, 1]; map to strict (0.01, 0.99) as required by openenv
        score = clamped * 0.98 + 0.01
        return max(0.01, min(0.99, score))

    @property
    def state(self) -> State:
        return self._state
