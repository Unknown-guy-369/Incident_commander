"""
Reward functions for the Incident Commander environment.

4 independent reward signals as required by the hackathon guidelines.
Each signal is kept independent — logged separately for monitoring.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List

if TYPE_CHECKING:
    pass  # avoid circular import

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_STEPS = 50
MIN_INVESTIGATION_STEPS = 3   # must read logs OR metrics before any fix

CORRECT_FIX_FOR = {
    "memory_limit_too_low":      "scale_up",
    "bad_deployment":            "rollback",
    "connection_pool_exhausted": "hotfix",
    "traffic_spike":             "scale_up",
    "dependency_failure":        "restart_pod",
    "config_error":              "hotfix",
    "redis_down":                "restart_pod",
    "certificate_expired":       "hotfix",
}


# ---------------------------------------------------------------------------
# Reward 1 — Service Recovery (independent health check)
# ---------------------------------------------------------------------------

def reward_service_recovery(post_fix_status: str | None) -> float:
    """Did the service actually recover after the fix? Verified by env state."""
    if post_fix_status == "recovered":
        return 30.0
    elif post_fix_status == "degraded":
        return 5.0
    elif post_fix_status == "worse":
        return -15.0
    else:
        return -20.0  # no fix ever applied


# ---------------------------------------------------------------------------
# Reward 2 — Root Cause Accuracy
# ---------------------------------------------------------------------------

def reward_root_cause_accuracy(
    locked_hypothesis: str | None,
    true_root_cause: str,
    actions_taken: List[str],
) -> float:
    """Did agent identify the correct root cause BEFORE taking fixing action?"""

    fix_actions = {"restart_pod", "rollback", "scale_up", "hotfix"}
    investigation_actions = {"read_logs", "read_metrics", "read_deployment_history", "read_dependency_graph"}

    if locked_hypothesis is None:
        # Never locked a hypothesis
        return -15.0

    # Check that identify_cause came before the first fix action
    try:
        hyp_idx   = next(i for i, a in enumerate(actions_taken) if a == "identify_cause")
        first_fix = next((i for i, a in enumerate(actions_taken) if a in fix_actions), len(actions_taken))
        if hyp_idx > first_fix:
            return -10.0  # fixed before hypothesising
    except StopIteration:
        pass

    if locked_hypothesis == true_root_cause:
        return 25.0
    return -10.0


# ---------------------------------------------------------------------------
# Reward 3 — Action Quality (anti-shortcuts)
# ---------------------------------------------------------------------------

def reward_action_quality(
    actions_taken: List[str],
    locked_hypothesis: str | None,
    true_root_cause: str,
    rollback_count: int,
    escalation_count: int,
    already_read_logs: set,
    already_read_metrics: set,
) -> float:

    penalty = 0.0
    investigation = {"read_logs", "read_metrics", "read_deployment_history", "read_dependency_graph"}
    fix_actions   = {"restart_pod", "rollback", "scale_up", "hotfix"}

    # Anti-shortcut 1: must investigate before fixing
    investigation_steps = sum(1 for a in actions_taken if a in investigation)
    first_fix_idx = next((i for i, a in enumerate(actions_taken) if a in fix_actions), None)
    if first_fix_idx is not None and investigation_steps < MIN_INVESTIGATION_STEPS:
        penalty -= 15.0

    # Anti-shortcut 2: fix must match hypothesis
    if locked_hypothesis and true_root_cause:
        expected_fix = CORRECT_FIX_FOR.get(locked_hypothesis)
        actual_fixes = [a for a in actions_taken if a in fix_actions]
        if actual_fixes and expected_fix and actual_fixes[0] != expected_fix:
            penalty -= 20.0  # wrong fix for identified cause

    # Anti-shortcut 3: rollback spam
    if rollback_count > 3:
        penalty -= 50.0
    elif rollback_count > 1 and true_root_cause != "bad_deployment":
        penalty -= 15.0

    # Anti-shortcut 4: escalation spam
    if escalation_count > 2:
        penalty -= 30.0

    # Anti-shortcut 5: resolve without reading anything
    if "resolve" in actions_taken and not already_read_logs and not already_read_metrics:
        penalty -= 25.0

    # Bonus: good investigation diversity
    if (already_read_logs and already_read_metrics):
        penalty += 5.0  # small bonus for complete investigation

    return penalty


# ---------------------------------------------------------------------------
# Reward 4 — Speed (only if service recovered)
# ---------------------------------------------------------------------------

def reward_speed(post_fix_status: str | None, steps_used: int) -> float:
    """Speed bonus — only meaningful if the fix actually worked."""
    if post_fix_status != "recovered":
        return 0.0  # speed irrelevant when not fixed
    remaining = max(0, MAX_STEPS - steps_used)
    return (remaining / MAX_STEPS) * 15.0


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

def compute_total_reward(
    post_fix_status: str | None,
    locked_hypothesis: str | None,
    true_root_cause: str,
    actions_taken: List[str],
    rollback_count: int,
    escalation_count: int,
    already_read_logs: set,
    already_read_metrics: set,
    steps_used: int,
) -> tuple[float, Dict[str, float]]:
    """Compute total reward and return a breakdown dict for logging."""

    r1 = reward_service_recovery(post_fix_status)
    r2 = reward_root_cause_accuracy(locked_hypothesis, true_root_cause, actions_taken)
    r3 = reward_action_quality(
        actions_taken, locked_hypothesis, true_root_cause,
        rollback_count, escalation_count,
        already_read_logs, already_read_metrics,
    )
    r4 = reward_speed(post_fix_status, steps_used)

    total = r1 + r2 + r3 + r4

    # Normalise to [0, 1]:
    #   max achievable raw score ≈ 30 + 25 + 5 + 15 = 75
    #   negative raw scores floor to 0.0
    MAX_RAW = 75.0
    clamped = max(0.0, min(1.0, total / MAX_RAW))

    breakdown = {
        "service_recovery":    r1,
        "root_cause_accuracy": r2,
        "action_quality":      r3,
        "speed":               r4,
        "total_raw":           total,
        "total_clamped":       clamped,
    }
    return clamped, breakdown
