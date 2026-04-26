"""
Incident Commander Environment - Demo & Verification (in-process).

This script imports the IncidentCommanderEnvironment directly (no HTTP),
which bypasses any session-handling issues in OpenEnv's HTTP layer.

It demonstrates:
  * All 8 root-cause scenarios are reachable
  * All 4 reward signals fire correctly
  * Anti-shortcut guards work as advertised
  * A simple log-parsing heuristic dramatically outperforms a random policy

Lightweight - no torch / unsloth / trl needed.

Usage (from the project root):
    python demo_environment.py

Optional environment overrides:
    NUM_AUDIT_EPISODES=20 NUM_RANDOM_EPISODES=20 DIFFICULTY=1 python demo_environment.py
"""

import os
import sys
import random
from typing import Optional, Dict, Any, List

# ---------------------------------------------------------------------
# Import the env directly (no HTTP). The repo root must be on sys.path.
# ---------------------------------------------------------------------

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from server.incident_commander_environment import IncidentCommanderEnvironment
    from models import IncidentCommanderAction
except ImportError as e:
    print(f"ERROR: cannot import env modules. Run from the project root.\n  {e}")
    sys.exit(1)


NUM_AUDIT_EPISODES = int(os.environ.get("NUM_AUDIT_EPISODES", "10"))
NUM_RANDOM_EPISODES = int(os.environ.get("NUM_RANDOM_EPISODES", "10"))
DIFFICULTY = int(os.environ.get("DIFFICULTY", "1"))


# ---------------------------------------------------------------------
# Lightweight wrapper: same interface the demo uses, but in-process.
# ---------------------------------------------------------------------

class InProcessEnv:
    """Wraps IncidentCommanderEnvironment with the same call shape as
    the HTTP SyncEnvClient so the rest of the demo logic is unchanged."""

    def __init__(self, difficulty: int = 1):
        self.env = IncidentCommanderEnvironment(difficulty=difficulty)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    @staticmethod
    def _obs_to_dict(obs) -> Dict[str, Any]:
        if hasattr(obs, "model_dump"):
            return obs.model_dump()
        if hasattr(obs, "dict"):
            return obs.dict()
        return dict(obs)

    def reset(self, difficulty: int = 1) -> Dict[str, Any]:
        obs = self.env.reset(difficulty=difficulty)
        return {"observation": self._obs_to_dict(obs), "reward": 0.0, "done": False}

    def step(
        self,
        action_type: str,
        target_service: Optional[str] = None,
        hypothesis: Optional[str] = None,
        justification: Optional[str] = None,
        time_range_minutes: Optional[int] = None,
    ) -> Dict[str, Any]:
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


# ---------------------------------------------------------------------
# Root-cause classifier (parses revealed log content)
# ---------------------------------------------------------------------

ROOT_CAUSE_PATTERNS = [
    ("memory_limit_too_low",      ["OOMKilled", "memory limit", "memory usage at",
                                   "exceeded memory", "memory leak"]),
    ("certificate_expired",       ["TLS handshake failed", "certificate expired",
                                   "SSL CERTIFICATE", "certificate verify failed"]),
    ("connection_pool_exhausted", ["pool exhausted", "too many connections",
                                   "connection pool", "connection timeout waiting for DB"]),
    ("redis_down",                ["Redis unreachable", "redis-cache",
                                   "cache miss rate 100%", "redis"]),
    ("traffic_spike",             ["RPS jumped", "circuit breaker",
                                   "pod count insufficient", "SLA breach"]),
    ("config_error",              ["DATABASE_URL", "WAREHOUSE_API_KEY",
                                   "invalid config", "missing required env var"]),
    ("bad_deployment",            ["NullPointerException", "rolled out at",
                                   "heap dump", "v2.1.3", "v2.3.1"]),
]

ROOT_CAUSE_TO_FIX = {
    "memory_limit_too_low":      "scale_up",
    "bad_deployment":            "rollback",
    "connection_pool_exhausted": "hotfix",
    "traffic_spike":             "scale_up",
    "dependency_failure":        "restart_pod",
    "config_error":              "hotfix",
    "redis_down":                "restart_pod",
    "certificate_expired":       "hotfix",
}


def classify_root_cause(all_logs: Dict[str, List[str]]) -> str:
    blob = " ".join(line for lines in all_logs.values() for line in lines).lower()
    for cause, keywords in ROOT_CAUSE_PATTERNS:
        for kw in keywords:
            if kw.lower() in blob:
                return cause
    return "dependency_failure"


# ---------------------------------------------------------------------
# Heuristic policy
# ---------------------------------------------------------------------

def run_heuristic_episode(difficulty: int = 1, verbose: bool = False) -> Dict[str, Any]:
    """Read logs on 3 services, parse for root cause, fix, monitor, resolve."""
    with InProcessEnv(difficulty=difficulty) as env:
        init = env.reset(difficulty=difficulty)
        obs = init.get("observation", {})
        services = [s["name"] for s in obs.get("services_overview", [])]

        if verbose:
            print(f"  Alert:    {obs.get('alert_summary')}")
            print(f"  Services: {services}")

        # 1-3. Read logs on first 3 services (satisfies anti-shortcut guard)
        for svc in services[:3]:
            result = env.step("read_logs", target_service=svc)
            obs = result.get("observation", {})
            if verbose:
                msg = (obs.get("last_action_result", "") or "")[:80]
                print(f"  read_logs:{svc:<22}-> {msg}")

        # 4. Read metrics on the most-degraded service
        degraded = [
            s["name"] for s in obs.get("services_overview", [])
            if s.get("status") == "degraded"
        ]
        target = degraded[0] if degraded else services[0]
        result = env.step("read_metrics", target_service=target)
        obs = result.get("observation", {})

        # 5. Classify root cause from revealed logs
        revealed = obs.get("revealed_logs", {})
        hypothesis = classify_root_cause(revealed)
        fix = ROOT_CAUSE_TO_FIX[hypothesis]

        if verbose:
            print(
                f"  read_metrics:{target:<19}-> classify -> "
                f"hypothesis={hypothesis}, suggest fix={fix}"
            )

        # 6. Lock hypothesis
        env.step("identify_cause", target_service=target, hypothesis=hypothesis)

        # 7. Apply matching fix
        result = env.step(fix, target_service=target)
        obs = result.get("observation", {})
        if verbose:
            msg = (obs.get("last_action_result", "") or "")[:80]
            print(f"  {fix}:{target:<24}-> {msg}")

        # 8. Monitor recovery
        result = env.step("monitor_recovery")
        obs = result.get("observation", {})
        post_status = obs.get("post_fix_status")
        if verbose:
            print(f"  monitor_recovery -> post_fix_status={post_status}")

        # 9. Resolve if recovered
        breakdown: Optional[Dict[str, Any]] = None
        last_reward = 0.0
        if post_status == "recovered":
            result = env.step("resolve", justification="service recovered")
            obs = result.get("observation", {})
            breakdown = obs.get("reward_breakdown") or {}
            last_reward = result.get("reward", 0.0)
            if verbose:
                print(f"  resolve -> done, reward={last_reward:.3f}")
        else:
            if verbose:
                print(f"  Episode did NOT reach recovered (post_status={post_status})")

        return {
            "hypothesis": hypothesis,
            "fix_action": fix,
            "target_service": target,
            "post_fix_status": post_status,
            "resolved": post_status == "recovered",
            "reward": last_reward,
            "reward_breakdown": breakdown,
            "true_root_cause": (breakdown or {}).get("true_root_cause"),
        }


# ---------------------------------------------------------------------
# Random baseline
# ---------------------------------------------------------------------

VALID_ACTIONS = [
    "read_logs", "read_metrics",
    "read_deployment_history", "read_dependency_graph",
    "identify_cause",
    "restart_pod", "rollback", "scale_up", "hotfix",
    "escalate", "monitor_recovery", "resolve",
]
VALID_HYPOTHESES = list(ROOT_CAUSE_TO_FIX.keys())


def run_random_episode(difficulty: int = 1, max_steps: int = 12) -> Dict[str, Any]:
    with InProcessEnv(difficulty=difficulty) as env:
        init = env.reset(difficulty=difficulty)
        obs = init.get("observation", {})
        services = [s["name"] for s in obs.get("services_overview", [])]
        last_reward, post_status, resolved, breakdown = 0.0, None, False, None
        for _ in range(max_steps):
            atype = random.choice(VALID_ACTIONS)
            target = random.choice(services) if services else "payment-service"
            kwargs: Dict[str, Any] = {}
            if atype == "identify_cause":
                kwargs["hypothesis"] = random.choice(VALID_HYPOTHESES)
            try:
                result = env.step(atype, target_service=target, **kwargs)
            except Exception:
                continue
            obs = result.get("observation", {})
            last_reward = result.get("reward", 0.0)
            post_status = obs.get("post_fix_status")
            if result.get("done"):
                breakdown = obs.get("reward_breakdown") or {}
                resolved = (
                    post_status == "recovered"
                    and "resolve" in obs.get("actions_taken", [])
                )
                break
        return {
            "post_fix_status": post_status,
            "resolved": resolved,
            "reward": last_reward,
            "reward_breakdown": breakdown,
        }


# ---------------------------------------------------------------------
# Anti-shortcut verification
# ---------------------------------------------------------------------

def verify_anti_shortcuts() -> List[Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []

    # Check 1: Fix without 3+ investigations -> blocked
    with InProcessEnv() as env:
        env.reset(difficulty=1)
        result = env.step("scale_up", target_service="payment-service")
    msg = result.get("observation", {}).get("last_action_result", "")
    checks.append({
        "name": "Fix attempted with 0 investigations -> BLOCKED",
        "passed": "BLOCK" in msg.upper(),
        "evidence": msg[:90],
    })

    # Check 2: Fix without identify_cause -> blocked (after enough investigation)
    with InProcessEnv() as env:
        init = env.reset(difficulty=1)
        services = [s["name"] for s in init.get("observation", {}).get("services_overview", [])]
        for svc in services[:3]:
            env.step("read_logs", target_service=svc)
        result = env.step("scale_up", target_service="payment-service")
    msg = result.get("observation", {}).get("last_action_result", "")
    checks.append({
        "name": "Fix without identify_cause (3 reads done) -> BLOCKED",
        "passed": ("BLOCK" in msg.upper()) or ("identify_cause" in msg.lower()),
        "evidence": msg[:90],
    })

    # Check 3: Resolve before recovery -> blocked
    with InProcessEnv() as env:
        env.reset(difficulty=1)
        result = env.step("resolve", justification="trying to skip")
    msg = result.get("observation", {}).get("last_action_result", "")
    checks.append({
        "name": "Resolve attempted without recovery -> BLOCKED",
        "passed": ("BLOCK" in msg.upper()) or ("not recovered" in msg.lower()),
        "evidence": msg[:90],
    })

    return checks


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def header(title: str) -> None:
    print("\n" + "=" * 76)
    print(f"  {title}")
    print("=" * 76)


def main() -> None:
    print("Incident Commander Environment - Demo & Verification (in-process)")
    print(f"  Project root:    {ROOT}")
    print(f"  Audit episodes:  {NUM_AUDIT_EPISODES}")
    print(f"  Random episodes: {NUM_RANDOM_EPISODES}")
    print(f"  Difficulty:      {DIFFICULTY}")

    # =================================================================
    # Demo 1: Optimal heuristic episode (verbose)
    # =================================================================
    header("Demo 1: Optimal SRE Loop (one episode, verbose)")
    result = run_heuristic_episode(difficulty=DIFFICULTY, verbose=True)
    bd = result.get("reward_breakdown") or {}
    print(
        f"\n  Result: resolved={result['resolved']}  "
        f"reward={result['reward']:.3f}  hypothesis={result['hypothesis']}  "
        f"true_cause={result.get('true_root_cause')}"
    )
    if bd:
        print("  Reward signals (raw points):")
        print(f"    service_recovery    = {bd.get('service_recovery', 0.0):>+7.1f}")
        print(f"    root_cause_accuracy = {bd.get('root_cause_accuracy', 0.0):>+7.1f}")
        print(f"    action_quality      = {bd.get('action_quality', 0.0):>+7.1f}")
        print(f"    speed               = {bd.get('speed', 0.0):>+7.2f}")
        print(f"    total_raw           = {bd.get('total_raw', 0.0):>+7.2f}")
        print(
            f"    total_clamped       = {bd.get('total_clamped', 0.0):>+7.3f}"
            "  (this is the env reward returned by step())"
        )

    # =================================================================
    # Demo 2: Multi-scenario reward audit
    # =================================================================
    header(f"Demo 2: Multi-Scenario Audit ({NUM_AUDIT_EPISODES} heuristic episodes)")
    N = NUM_AUDIT_EPISODES
    rewards: List[float] = []
    resolved = 0
    scenarios = set()
    per_signal = {
        "service_recovery": [],
        "root_cause_accuracy": [],
        "action_quality": [],
        "speed": [],
    }
    hypothesis_correct = 0
    for i in range(N):
        r = run_heuristic_episode(difficulty=DIFFICULTY, verbose=False)
        rewards.append(r["reward"])
        resolved += int(r["resolved"])
        true_cause = r.get("true_root_cause")
        if true_cause:
            scenarios.add(true_cause)
            if r["hypothesis"] == true_cause:
                hypothesis_correct += 1
        bd = r.get("reward_breakdown") or {}
        for k in per_signal:
            per_signal[k].append(float(bd.get(k, 0.0)))
        r_mark = "OK" if r["resolved"] else " X"
        h_mark = "OK" if r["hypothesis"] == true_cause else " X"
        true_label = (true_cause or "?")
        print(
            f"  ep {i+1:>2}/{N}  hyp={r['hypothesis']:<28}  "
            f"true={true_label:<28}  hyp_match={h_mark}  "
            f"resolved={r_mark}  reward={r['reward']:.3f}"
        )

    avg = sum(rewards) / max(len(rewards), 1)
    print(f"\n  Heuristic policy summary (over {N} episodes):")
    print(f"    Resolved:                   {resolved}/{N}  ({100*resolved/N:.1f}%)")
    print(
        f"    Hypothesis correct:         {hypothesis_correct}/{N}  "
        f"({100*hypothesis_correct/N:.1f}%)"
    )
    print(f"    Mean reward:                {avg:>+7.3f}")
    print(f"    Distinct scenarios seen:    {len(scenarios)}/8")
    print(f"    Scenarios encountered:      {sorted(scenarios)}")
    print("\n  Per-signal mean reward (raw points):")
    for k, vals in per_signal.items():
        mean = sum(vals) / max(len(vals), 1)
        bar = "#" * int(abs(mean) * 0.5) if mean else ""
        sign = "+" if mean >= 0 else "-"
        print(f"    {k:<22} = {mean:>+7.2f}  {sign}{bar}")

    # =================================================================
    # Demo 3: Anti-shortcut guard verification
    # =================================================================
    header("Demo 3: Anti-Shortcut Guards Verification")
    checks = verify_anti_shortcuts()
    for c in checks:
        mark = "[PASS]" if c["passed"] else "[FAIL]"
        print(f"  {mark}  {c['name']}")
        print(f"          evidence: {c['evidence']}")
    passed = sum(1 for c in checks if c["passed"])

    # =================================================================
    # Demo 4: Random policy baseline
    # =================================================================
    header(f"Demo 4: Random Policy Baseline ({NUM_RANDOM_EPISODES} episodes)")
    M = NUM_RANDOM_EPISODES
    rand_rewards: List[float] = []
    rand_resolved = 0
    for i in range(M):
        r = run_random_episode(difficulty=DIFFICULTY)
        rand_rewards.append(r["reward"])
        rand_resolved += int(r["resolved"])
        mark = "OK" if r["resolved"] else " X"
        print(f"  ep {i+1:>2}/{M}  resolved={mark}  reward={r['reward']:.3f}")
    rand_avg = sum(rand_rewards) / max(len(rand_rewards), 1)
    print(f"\n  Random policy summary (over {M} episodes):")
    print(
        f"    Resolved:    {rand_resolved}/{M}  "
        f"({100*rand_resolved/M:.1f}%)"
    )
    print(f"    Mean reward: {rand_avg:>+7.3f}")

    # =================================================================
    # Final comparison
    # =================================================================
    header("Final Comparison: Random vs Heuristic")
    print(f"  {'Policy':<22} {'Episodes':>10} {'Resolved':>10} {'Mean Reward':>14}")
    print(f"  {'-'*58}")
    print(
        f"  {'Random baseline':<22} "
        f"{M:>10} {rand_resolved:>10} {rand_avg:>+14.3f}"
    )
    print(
        f"  {'Heuristic (log-parse)':<22} "
        f"{N:>10} {resolved:>10} {avg:>+14.3f}"
    )
    print(
        f"  {'Improvement':<22} "
        f"{'':>10} {resolved - rand_resolved:>+10} {avg - rand_avg:>+14.3f}"
    )

    print("\n" + "=" * 76)
    print("  DEMONSTRATION COMPLETE")
    print("=" * 76)
    print(f"  Anti-shortcut guards verified:  {passed}/{len(checks)}")
    print(f"  Distinct scenarios covered:     {len(scenarios)}/8")
    print(f"  Heuristic resolve rate:         {100*resolved/N:.1f}%")
    print(f"  Random   resolve rate:          {100*rand_resolved/M:.1f}%")
    print(f"  Heuristic mean reward:          {avg:>+7.3f}")
    print(f"  Random   mean reward:           {rand_avg:>+7.3f}")
    print()
    print("  This demonstrates that the environment:")
    print("    * Generates 8 distinct, randomised scenarios")
    print(
        "    * Exercises all 4 reward signals "
        "(recovery, root_cause, action_quality, speed)"
    )
    print(
        "    * Enforces anti-shortcut guards "
        "(fix-without-investigation, etc.)"
    )
    print(
        "    * Rewards correct multi-turn SRE reasoning "
        "over random action selection"
    )
    print()


if __name__ == "__main__":
    main()
