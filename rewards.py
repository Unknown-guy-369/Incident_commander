"""
Reward functions for the Incident Commander environment.

4 independent reward signals as required by the hackathon guidelines.
Each signal is kept independent and is also exposed as its own
GRPO-compatible callable so the trainer can log them separately.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Sequence

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_STEPS = 50
MIN_INVESTIGATION_STEPS = 3   # must read logs OR metrics before any fix

# Maximum theoretical raw aggregate (used for [0,1] clamping).
MAX_RAW = 30.0 + 25.0 + 5.0 + 15.0  # = 75.0

# Approximate minimum (used for [-1,1] signed clamping).
MIN_RAW_SIGNED = -75.0

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
# Reward 1 - Service Recovery
# ---------------------------------------------------------------------------

def reward_service_recovery(post_fix_status):
    """Did the service actually recover after the fix?"""
    if post_fix_status == "recovered":
        return 30.0
    elif post_fix_status == "degraded":
        return 5.0
    elif post_fix_status == "worse":
        return -15.0
    else:
        return -20.0


# ---------------------------------------------------------------------------
# Reward 2 - Root Cause Accuracy
# ---------------------------------------------------------------------------

def reward_root_cause_accuracy(locked_hypothesis, true_root_cause, actions_taken):
    """Did the agent identify the correct root cause BEFORE fixing?"""
    fix_actions = {"restart_pod", "rollback", "scale_up", "hotfix"}

    if locked_hypothesis is None:
        return -15.0

    try:
        hyp_idx = next(i for i, a in enumerate(actions_taken) if a == "identify_cause")
        first_fix = next(
            (i for i, a in enumerate(actions_taken) if a in fix_actions),
            len(actions_taken),
        )
        if hyp_idx > first_fix:
            return -10.0
    except StopIteration:
        pass

    if locked_hypothesis == true_root_cause:
        return 25.0
    return -10.0


# ---------------------------------------------------------------------------
# Reward 3 - Action Quality (anti-shortcuts)
# ---------------------------------------------------------------------------

def reward_action_quality(
    actions_taken,
    locked_hypothesis,
    true_root_cause,
    rollback_count,
    escalation_count,
    already_read_logs,
    already_read_metrics,
):
    penalty = 0.0
    investigation = {"read_logs", "read_metrics", "read_deployment_history", "read_dependency_graph"}
    fix_actions = {"restart_pod", "rollback", "scale_up", "hotfix"}

    investigation_steps = sum(1 for a in actions_taken if a in investigation)
    first_fix_idx = next((i for i, a in enumerate(actions_taken) if a in fix_actions), None)
    if first_fix_idx is not None and investigation_steps < MIN_INVESTIGATION_STEPS:
        penalty -= 15.0

    if locked_hypothesis and true_root_cause:
        expected_fix = CORRECT_FIX_FOR.get(locked_hypothesis)
        actual_fixes = [a for a in actions_taken if a in fix_actions]
        if actual_fixes and expected_fix and actual_fixes[0] != expected_fix:
            penalty -= 20.0

    if rollback_count > 3:
        penalty -= 50.0
    elif rollback_count > 1 and true_root_cause != "bad_deployment":
        penalty -= 15.0

    if escalation_count > 2:
        penalty -= 30.0

    if "resolve" in actions_taken and not already_read_logs and not already_read_metrics:
        penalty -= 25.0

    if already_read_logs and already_read_metrics:
        penalty += 5.0

    return penalty


# ---------------------------------------------------------------------------
# Reward 4 - Speed
# ---------------------------------------------------------------------------

def reward_speed(post_fix_status, steps_used):
    """Speed bonus - only counts if the fix actually worked."""
    if post_fix_status != "recovered":
        return 0.0
    remaining = max(0, MAX_STEPS - steps_used)
    return (remaining / MAX_STEPS) * 15.0


# ---------------------------------------------------------------------------
# Aggregator (legacy, [0,1] clamped - keeps existing tests green)
# ---------------------------------------------------------------------------

def compute_total_reward(
    post_fix_status,
    locked_hypothesis,
    true_root_cause,
    actions_taken,
    rollback_count,
    escalation_count,
    already_read_logs,
    already_read_metrics,
    steps_used,
):
    """Compute total reward + breakdown dict, clamped to [0, 1]."""
    r1 = reward_service_recovery(post_fix_status)
    r2 = reward_root_cause_accuracy(locked_hypothesis, true_root_cause, actions_taken)
    r3 = reward_action_quality(
        actions_taken, locked_hypothesis, true_root_cause,
        rollback_count, escalation_count,
        already_read_logs, already_read_metrics,
    )
    r4 = reward_speed(post_fix_status, steps_used)

    total = r1 + r2 + r3 + r4
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


def compute_signed_reward(
    post_fix_status,
    locked_hypothesis,
    true_root_cause,
    actions_taken,
    rollback_count,
    escalation_count,
    already_read_logs,
    already_read_metrics,
    steps_used,
):
    """Same components as compute_total_reward but signed in roughly [-1, 1]."""
    r1 = reward_service_recovery(post_fix_status)
    r2 = reward_root_cause_accuracy(locked_hypothesis, true_root_cause, actions_taken)
    r3 = reward_action_quality(
        actions_taken, locked_hypothesis, true_root_cause,
        rollback_count, escalation_count,
        already_read_logs, already_read_metrics,
    )
    r4 = reward_speed(post_fix_status, steps_used)

    total = r1 + r2 + r3 + r4
    if total >= 0:
        signed = total / MAX_RAW
    else:
        signed = total / abs(MIN_RAW_SIGNED)
    signed = max(-1.0, min(1.0, signed))

    breakdown = {
        "service_recovery":    r1,
        "root_cause_accuracy": r2,
        "action_quality":      r3,
        "speed":               r4,
        "total_raw":           total,
        "total_signed":        signed,
    }
    return signed, breakdown


# ---------------------------------------------------------------------------
# Shaping bonus
# ---------------------------------------------------------------------------

def shaping_bonus(actions_taken, read_root_service):
    """Tiny dense bonus to keep gradients alive."""
    bonus = 0.0
    if "read_logs" in actions_taken or "read_metrics" in actions_taken:
        bonus += 0.05
    if read_root_service:
        bonus += 0.05
    if "identify_cause" in actions_taken:
        bonus += 0.05
    if "monitor_recovery" in actions_taken:
        bonus += 0.05
    return bonus


# ---------------------------------------------------------------------------
# Format reward + parser
# ---------------------------------------------------------------------------

_THOUGHT_RE = re.compile(r"<thought>.*?</thought>", re.DOTALL)
_ACTION_RE = re.compile(r"<action>\s*([^:<\n]+?)\s*(?::\s*([^<\n]*?))?\s*</action>", re.DOTALL)


def parse_action(completion):
    """Parse <action>action_type:target</action>."""
    m = _ACTION_RE.search(completion)
    if not m:
        return None, None
    atype = m.group(1).strip()
    target = m.group(2).strip() if m.group(2) else ""
    return atype, target


def format_reward_score(completion):
    """Static format check - returns a float in [0.0, 0.2]."""
    score = 0.0
    if _THOUGHT_RE.search(completion):
        score += 0.1
    if _ACTION_RE.search(completion):
        score += 0.1
    return score


# ---------------------------------------------------------------------------
# GRPO-compatible reward function factory
# ---------------------------------------------------------------------------

def make_grpo_reward_fns(rollout_fn):
    """Build TRL-compatible reward functions out of a rollout callable.

    rollout_fn(prompt, completion) must return a dict with the EpisodeState
    fields (see rollout.py). Internally we memoise rollouts by (id(prompt),
    id(completion)) so each physical episode is rolled out exactly once per
    training step, even though the reward functions are called in sequence.
    """
    _cache = {"keys": None, "data": {}}

    _empty = {
        "post_fix_status": None,
        "locked_hypothesis": None,
        "true_root_cause": "",
        "actions_taken": [],
        "rollback_count": 0,
        "escalation_count": 0,
        "already_read_logs": set(),
        "already_read_metrics": set(),
        "steps_used": 0,
        "read_root_service": False,
    }

    def _gather_states(prompts, completions):
        keys = tuple((id(p), id(c)) for p, c in zip(prompts, completions))
        if _cache["keys"] != keys:
            _cache["data"] = {}
            _cache["keys"] = keys
        out = []
        for k, p, c in zip(keys, prompts, completions):
            if k not in _cache["data"]:
                try:
                    _cache["data"][k] = rollout_fn(p, c)
                except Exception as e:
                    rec = dict(_empty)
                    rec["_error"] = str(e)
                    _cache["data"][k] = rec
            out.append(_cache["data"][k])
        return out

    def recovery_fn(prompts, completions, **kwargs):
        states = _gather_states(prompts, completions)
        return [reward_service_recovery(s["post_fix_status"]) / 30.0 for s in states]

    def root_cause_fn(prompts, completions, **kwargs):
        states = _gather_states(prompts, completions)
        return [
            reward_root_cause_accuracy(
                s["locked_hypothesis"], s["true_root_cause"], s["actions_taken"]
            ) / 25.0
            for s in states
        ]

    def action_quality_fn(prompts, completions, **kwargs):
        states = _gather_states(prompts, completions)
        return [
            reward_action_quality(
                s["actions_taken"], s["locked_hypothesis"], s["true_root_cause"],
                s["rollback_count"], s["escalation_count"],
                s["already_read_logs"], s["already_read_metrics"],
            ) / 50.0
            for s in states
        ]

    def speed_fn(prompts, completions, **kwargs):
        states = _gather_states(prompts, completions)
        return [
            reward_speed(s["post_fix_status"], s["steps_used"]) / 15.0
            for s in states
        ]

    def shaping_fn(prompts, completions, **kwargs):
        states = _gather_states(prompts, completions)
        return [shaping_bonus(s["actions_taken"], s["read_root_service"]) for s in states]

    def format_fn(prompts, completions, **kwargs):
        return [format_reward_score(c) for c in completions]

    recovery_fn.__name__ = "reward_recovery"
    root_cause_fn.__name__ = "reward_root_cause"
    action_quality_fn.__name__ = "reward_action_quality"
    speed_fn.__name__ = "reward_speed"
    shaping_fn.__name__ = "reward_shaping"
    format_fn.__name__ = "reward_format"

    return [recovery_fn, root_cause_fn, action_quality_fn, speed_fn,
            shaping_fn, format_fn]
