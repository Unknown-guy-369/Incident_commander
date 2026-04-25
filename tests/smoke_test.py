"""
Standalone smoke test — no pytest required.
Run with: python tests/smoke_test.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ..models import IncidentCommanderAction
from ..simulator import ALL_SCENARIOS, CORRECT_FIX, VALID_ROOT_CAUSES, Simulator
from ..rewards import compute_total_reward, MIN_INVESTIGATION_STEPS
from ..server.incident_commander_environment import IncidentCommanderEnvironment

PASS = "[PASS]"
FAIL = "[FAIL]"
results = []

def check(name, condition, detail=""):
    mark = PASS if condition else FAIL
    msg = f"{mark} {name}"
    if not condition and detail:
        msg += f"\n     → {detail}"
    print(msg)
    results.append(condition)

def act(env, action_type, target_service="payment-service", **kwargs):
    return env.step(IncidentCommanderAction(
        action_type=action_type, target_service=target_service, **kwargs
    ))

def read_enough(env):
    svcs = [s["name"] for s in env._obs().services_overview]
    for svc in svcs[:MIN_INVESTIGATION_STEPS]:
        act(env, "read_logs", svc)

# --- Test 1: Reset stability --------------------------------------------------
print("\n=== Test 1: Reset Stability ===")
env = IncidentCommanderEnvironment(difficulty=1)
ok = True
for i in range(20):
    obs = env.reset()
    if obs.steps_remaining != 50 or obs.done or obs.alert_summary == "" or obs.revealed_logs != {}:
        ok = False
        break
check("20 resets all give clean state", ok)

# --- Test 2: Partial observability --------------------------------------------
print("\n=== Test 2: Partial Observability ===")
env = IncidentCommanderEnvironment(difficulty=1)
env.reset()
obs = act(env, "read_metrics", "payment-service")
check("read_metrics does not reveal logs", "payment-service" not in obs.revealed_logs)
obs2 = act(env, "read_logs", "payment-service")
check("read_logs reveals that service's logs", "payment-service" in obs2.revealed_logs)
check("deployment history hidden until requested", obs2.deployment_history is None)
obs3 = act(env, "read_deployment_history", "payment-service")
check("deployment history revealed after request", obs3.deployment_history is not None)
obs4 = act(env, "read_dependency_graph", "payment-service")
check("dependency graph revealed after request", obs4.dependency_graph is not None)

# --- Test 3: Anti-shortcuts ---------------------------------------------------
print("\n=== Test 3: Anti-Shortcuts ===")
env = IncidentCommanderEnvironment(difficulty=1)
env.reset()
# Too few investigation steps
act(env, "read_logs", "payment-service")
obs = act(env, "restart_pod", "payment-service")
check("Fix blocked with <3 investigation steps", "BLOCKED" in obs.last_action_result)

env2 = IncidentCommanderEnvironment(difficulty=1)
env2.reset()
read_enough(env2)
obs = act(env2, "restart_pod", "payment-service")
check("Fix blocked without hypothesis", "BLOCKED" in obs.last_action_result)

env3 = IncidentCommanderEnvironment(difficulty=1)
env3.reset()
obs = act(env3, "resolve", "payment-service")
check("Resolve blocked before recovery", obs.done is False)

# --- Test 4: Correct episode --------------------------------------------------
print("\n=== Test 4: Correct Episode Flow ===")
rewards = []
for trial in range(5):
    env = IncidentCommanderEnvironment(difficulty=1)
    env.reset()
    sc = env._ctx.scenario
    svcs = [s["name"] for s in env._obs().services_overview]
    for svc in svcs[:MIN_INVESTIGATION_STEPS]:
        act(env, "read_logs", svc)
    act(env, "read_deployment_history", sc.root_service)
    act(env, "read_dependency_graph", sc.root_service)
    act(env, "identify_cause", sc.root_service, hypothesis=sc.root_cause)
    correct_fix = CORRECT_FIX[sc.root_cause]
    act(env, correct_fix, sc.root_service)
    act(env, "monitor_recovery", sc.root_service)
    obs = act(env, "resolve", sc.root_service)
    rewards.append(obs.reward)
check("5 correct episodes all done=True", all(r is not None for r in rewards))
all_positive = all(r > 0 for r in rewards)
check(f"5 correct episodes all positive reward: {[round(r,3) for r in rewards]}", all_positive)

# --- Test 5: Timeout ----------------------------------------------------------
print("\n=== Test 5: Timeout ===")
env = IncidentCommanderEnvironment(difficulty=1)
env.reset()
obs = None
for _ in range(50):
    obs = act(env, "read_logs", "payment-service")
check("Episode done after 50 steps", obs.done is True)
check("Timeout gives 0.0 reward (floored)", obs.reward == 0.0)

# --- Test 6: Multi-round fix loop ---------------------------------------------
print("\n=== Test 6: Multi-Round Fix Loop ===")
env = IncidentCommanderEnvironment(difficulty=1)
env.reset()
sc = env._ctx.scenario
svcs = [s["name"] for s in env._obs().services_overview]
for svc in svcs[:MIN_INVESTIGATION_STEPS]:
    act(env, "read_logs", svc)
wrong_cause_true_fix = CORRECT_FIX[sc.root_cause]
wrong_causes = [c for c in VALID_ROOT_CAUSES if c != sc.root_cause and CORRECT_FIX.get(c) != wrong_cause_true_fix]
wrong_hyp = wrong_causes[0]
wrong_fix = CORRECT_FIX[wrong_hyp]
act(env, "identify_cause", sc.root_service, hypothesis=wrong_hyp)
act(env, wrong_fix, sc.root_service)
obs = act(env, "monitor_recovery", sc.root_service)
check("Wrong fix gives degraded/worse (not recovered)", obs.post_fix_status in {"degraded", "worse"})

# Apply correct fix after observing degraded
correct_fix = CORRECT_FIX[sc.root_cause]
act(env, correct_fix, sc.root_service)
obs2 = act(env, "monitor_recovery", sc.root_service)
check("Correct fix after degraded gives recovered", obs2.post_fix_status == "recovered")

# --- Test 7: Scenario randomisation ------------------------------------------
print("\n=== Test 7: Scenario Randomisation ===")
check("All root causes are valid", all(sc.root_cause in VALID_ROOT_CAUSES for sc in ALL_SCENARIOS))
check("All correct fixes match CORRECT_FIX", all(CORRECT_FIX.get(sc.root_cause) == sc.correct_fix for sc in ALL_SCENARIOS))

sim = Simulator(difficulty=4)
seen_scenarios = set()
for _ in range(30):
    ctx = sim.sample()
    seen_scenarios.add(ctx.scenario.name)
check(f"Simulator covers diverse scenarios ({len(seen_scenarios)}/8 seen in 30 samples)", len(seen_scenarios) >= 4)

# --- Summary -----------------------------------------------------------------
print(f"\n{'='*50}")
passed = sum(results)
total = len(results)
print(f"Result: {passed}/{total} checks passed")
if passed == total:
    print("[PASS] ALL STABILITY CHECKS PASSED — safe to train")
else:
    print(f"[FAIL] {total - passed} checks FAILED — fix before training")
    sys.exit(1)
