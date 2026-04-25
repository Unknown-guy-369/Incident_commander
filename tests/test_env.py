"""
Stability tests for the Incident Commander environment.
Run with: pytest tests/test_env.py -v
"""

from __future__ import annotations

import sys
import os

# Allow running from the tests/ directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from models import IncidentCommanderAction, IncidentCommanderObservation
from simulator import ALL_SCENARIOS, CORRECT_FIX, VALID_ROOT_CAUSES, Simulator
from rewards import compute_total_reward, MIN_INVESTIGATION_STEPS
from server.incident_commander_environment import IncidentCommanderEnvironment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_env(difficulty: int = 1) -> IncidentCommanderEnvironment:
    return IncidentCommanderEnvironment(difficulty=difficulty)


def act(env, action_type, target_service="payment-service", **kwargs):
    return env.step(IncidentCommanderAction(
        action_type=action_type,
        target_service=target_service,
        **kwargs,
    ))


# ---------------------------------------------------------------------------
# Test 1: Reset stability
# ---------------------------------------------------------------------------

class TestResetStability:
    def test_reset_100_times(self):
        env = make_env()
        for i in range(100):
            obs = env.reset()
            assert isinstance(obs, IncidentCommanderObservation), f"reset {i} returned wrong type"
            assert obs.steps_remaining == 50, f"steps_remaining wrong on reset {i}"
            assert obs.hypothesis_locked is False
            assert obs.done is False
            assert obs.alert_summary != ""
            assert len(obs.services_overview) > 0

    def test_reset_returns_no_revealed_data(self):
        """After reset, agent should see no logs or metrics (partial observability)."""
        env = make_env()
        obs = env.reset()
        assert obs.revealed_logs == {}
        assert obs.revealed_metrics == {}
        assert obs.deployment_history is None
        assert obs.dependency_graph is None


# ---------------------------------------------------------------------------
# Test 2: Partial observability
# ---------------------------------------------------------------------------

class TestPartialObservability:
    def test_logs_not_revealed_until_requested(self):
        env = make_env()
        env.reset()
        obs = act(env, "read_metrics", "payment-service")
        # logs still empty
        assert "payment-service" not in obs.revealed_logs

    def test_read_logs_reveals_only_that_service(self):
        env = make_env()
        env.reset()
        obs = act(env, "read_logs", "payment-service")
        assert "payment-service" in obs.revealed_logs
        # Other services not yet revealed
        for svc, logs in obs.revealed_logs.items():
            assert svc == "payment-service", f"Unexpected service {svc} in revealed_logs"

    def test_deployment_history_hidden_until_requested(self):
        env = make_env()
        env.reset()
        obs = act(env, "read_logs", "payment-service")
        assert obs.deployment_history is None
        obs2 = act(env, "read_deployment_history", "payment-service")
        assert obs2.deployment_history is not None

    def test_dependency_graph_hidden_until_requested(self):
        env = make_env()
        env.reset()
        obs = act(env, "read_dependency_graph", "payment-service")
        assert obs.dependency_graph is not None


# ---------------------------------------------------------------------------
# Test 3: Anti-shortcuts
# ---------------------------------------------------------------------------

class TestAntiShortcuts:
    def _read_enough(self, env, services=None):
        """Helper: read logs+metrics for MIN_INVESTIGATION_STEPS distinct services."""
        obs = env._obs()
        all_services = [s["name"] for s in obs.services_overview]
        services = services or all_services
        reads = 0
        for svc in services:
            if reads >= MIN_INVESTIGATION_STEPS:
                break
            act(env, "read_logs", svc)
            reads += 1

    def test_fix_blocked_without_investigation(self):
        """Fix should be blocked if agent hasn't read enough sources."""
        env = make_env()
        env.reset()
        # Read only 1 source (below MIN_INVESTIGATION_STEPS=3)
        act(env, "read_logs", "payment-service")
        obs = act(env, "restart_pod", "payment-service")
        assert "BLOCKED" in obs.last_action_result
        assert obs.done is False  # not ended, just penalised

    def test_fix_blocked_without_hypothesis(self):
        """Fix should be blocked if agent never called identify_cause."""
        env = make_env()
        env.reset()
        self._read_enough(env)
        obs = act(env, "restart_pod", "payment-service")
        assert "BLOCKED" in obs.last_action_result

    def test_resolve_blocked_before_recovery(self):
        """resolve() should be rejected if service hasn't recovered."""
        env = make_env()
        env.reset()
        obs = act(env, "resolve", "payment-service")
        assert "BLOCKED" in obs.last_action_result or "not recovered" in obs.last_action_result.lower()
        assert obs.done is False

    def test_rollback_hard_cap(self):
        """After 4 rollbacks agent should be hard-stopped with heavy penalty."""
        env = make_env()
        env.reset()
        self._read_enough(env)
        act(env, "identify_cause", "payment-service", hypothesis="bad_deployment")
        for _ in range(4):
            obs = act(env, "rollback", "payment-service")
        assert obs.reward <= 0.0

    def test_escalation_hard_cap(self):
        env = make_env()
        env.reset()
        for _ in range(3):
            obs = act(env, "escalate", "payment-service",
                      justification="test escalation")
        assert obs.reward <= 0.0


# ---------------------------------------------------------------------------
# Test 4: Correct episode flow
# ---------------------------------------------------------------------------

class TestCorrectEpisodeFlow:
    def _correct_episode(self, env: IncidentCommanderEnvironment) -> IncidentCommanderObservation:
        """Play a correct episode and return the final obs."""
        env.reset()
        sc = env._ctx.scenario

        # Investigate enough services
        all_services = [s["name"] for s in env._obs().services_overview]
        reads = 0
        for svc in all_services:
            if reads >= MIN_INVESTIGATION_STEPS:
                break
            act(env, "read_logs", svc)
            reads += 1

        act(env, "read_deployment_history", sc.root_service)
        act(env, "read_dependency_graph", sc.root_service)

        # Lock correct hypothesis
        act(env, "identify_cause", sc.root_service,
            hypothesis=sc.root_cause)

        # Apply correct fix
        correct_fix = CORRECT_FIX[sc.root_cause]
        act(env, correct_fix, sc.root_service)

        # Observe recovery
        act(env, "monitor_recovery", sc.root_service)

        # Resolve
        obs = act(env, "resolve", sc.root_service)
        return obs

    def test_correct_episode_positive_reward(self):
        env = make_env()
        obs = self._correct_episode(env)
        assert obs.done is True, "Episode should be done after resolve"
        assert obs.reward > 0, f"Correct episode should have positive reward, got {obs.reward}"

    def test_correct_episode_reward_breakdown_populated(self):
        env = make_env()
        obs = self._correct_episode(env)
        assert obs.reward_breakdown is not None
        bd = obs.reward_breakdown
        assert "service_recovery" in bd
        assert "root_cause_accuracy" in bd
        assert "action_quality" in bd
        assert "speed" in bd
        assert bd["service_recovery"] > 0, "Correct episode: recovery reward should be positive"
        assert bd["root_cause_accuracy"] > 0, "Correct episode: root cause reward should be positive"

    def test_correct_episode_10_times(self):
        """All correct episodes should produce positive reward — consistency check."""
        env = make_env()
        rewards = []
        for _ in range(10):
            obs = self._correct_episode(env)
            rewards.append(obs.reward)
        assert all(r > 0 for r in rewards), f"Some correct episodes got non-positive reward: {rewards}"


# ---------------------------------------------------------------------------
# Test 5: Timeout fires correctly
# ---------------------------------------------------------------------------

class TestTimeout:
    def test_timeout_at_50_steps(self):
        env = make_env()
        env.reset()
        obs = None
        for _ in range(50):
            obs = act(env, "read_logs", "payment-service")
        assert obs.done is True, "Should be done after 50 steps"
        assert obs.reward == 0.0  # negative raw floored to 0.0

    def test_steps_remaining_counts_down(self):
        env = make_env()
        env.reset()
        for i in range(5):
            obs = act(env, "read_logs", "payment-service")
        assert obs.steps_remaining == 50 - 5  # after 5 steps, 45 remain


# ---------------------------------------------------------------------------
# Test 6: Simulator randomisation
# ---------------------------------------------------------------------------

class TestSimulatorRandomisation:
    def test_log_lines_differ_across_episodes(self):
        """Same scenario type, different episode → different log content (randomised values)."""
        env = make_env()
        results = set()
        for _ in range(5):
            sc_name = env.reset()  # re-seeds rng
            obs = act(env, "read_logs", env._ctx.scenario.root_service)
            lines_key = str(obs.revealed_logs)
            results.add(lines_key)
        # At least 2 different log outputs across 5 episodes
        assert len(results) >= 2, "Logs appear to be static across episodes (not random)"

    def test_scenario_bank_coverage(self):
        """Simulator can sample from all 8 scenarios without crash."""
        sim = Simulator(difficulty=4)
        seen = set()
        for _ in range(100):
            ctx = sim.sample()
            seen.add(ctx.scenario.name)
        assert len(seen) >= 4, f"Only saw {len(seen)} distinct scenarios in 100 samples"

    def test_all_root_causes_valid(self):
        """Every scenario's root cause is in VALID_ROOT_CAUSES."""
        for sc in ALL_SCENARIOS:
            assert sc.root_cause in VALID_ROOT_CAUSES, f"{sc.name}: '{sc.root_cause}' not valid"

    def test_all_correct_fixes_valid(self):
        """Every scenario's correct_fix matches CORRECT_FIX mapping."""
        for sc in ALL_SCENARIOS:
            assert CORRECT_FIX.get(sc.root_cause) == sc.correct_fix, (
                f"{sc.name}: fix mismatch — expected {CORRECT_FIX.get(sc.root_cause)}, "
                f"got {sc.correct_fix}"
            )


# ---------------------------------------------------------------------------
# Test 7: Multi-round fix loop
# ---------------------------------------------------------------------------

class TestMultiRoundFixLoop:
    def test_wrong_fix_gives_degraded_or_worse(self):
        env = make_env()
        env.reset()
        sc = env._ctx.scenario

        # Investigate
        for svc in [s["name"] for s in env._obs().services_overview][:MIN_INVESTIGATION_STEPS]:
            act(env, "read_logs", svc)

        # Deliberately wrong hypothesis + wrong fix
        correct_fix_true = CORRECT_FIX[sc.root_cause]
        wrong_causes = [c for c in VALID_ROOT_CAUSES if c != sc.root_cause and CORRECT_FIX.get(c) != correct_fix_true]
        wrong_hyp = wrong_causes[0]
        wrong_fix = CORRECT_FIX[wrong_hyp]

        act(env, "identify_cause", sc.root_service, hypothesis=wrong_hyp)
        act(env, wrong_fix, sc.root_service)
        obs = act(env, "monitor_recovery", sc.root_service)

        assert obs.post_fix_status in {"degraded", "worse"}, (
            f"Wrong fix should not lead to 'recovered', got: {obs.post_fix_status}"
        )

    def test_can_apply_second_fix_after_degraded(self):
        """Agent can apply a different fix after observing degraded status."""
        env = make_env()
        env.reset()
        sc = env._ctx.scenario

        for svc in [s["name"] for s in env._obs().services_overview][:MIN_INVESTIGATION_STEPS]:
            act(env, "read_logs", svc)

        # First wrong fix
        wrong_causes = [c for c in VALID_ROOT_CAUSES if c != sc.root_cause]
        act(env, "identify_cause", sc.root_service, hypothesis=wrong_causes[0])
        wrong_fix = CORRECT_FIX[wrong_causes[0]]
        act(env, wrong_fix, sc.root_service)
        act(env, "monitor_recovery", sc.root_service)

        # Now apply correct fix
        correct_fix = CORRECT_FIX[sc.root_cause]
        act(env, correct_fix, sc.root_service)
        obs = act(env, "monitor_recovery", sc.root_service)
        assert obs.post_fix_status == "recovered"


# ---------------------------------------------------------------------------
# Test 8: Reward functions
# ---------------------------------------------------------------------------

class TestRewardFunctions:
    def test_service_not_fixed_negative_reward(self):
        clamped, bd = compute_total_reward(
            post_fix_status=None,
            locked_hypothesis=None,
            true_root_cause="bad_deployment",
            actions_taken=["read_logs", "resolve"],
            rollback_count=0,
            escalation_count=0,
            already_read_logs={"api-gateway"},
            already_read_metrics=set(),
            steps_used=5,
        )
        assert bd["service_recovery"] < 0  # raw score still negative
        assert bd["root_cause_accuracy"] < 0
        assert clamped == 0.0  # floored at 0

    def test_perfect_episode_high_reward(self):
        clamped, bd = compute_total_reward(
            post_fix_status="recovered",
            locked_hypothesis="bad_deployment",
            true_root_cause="bad_deployment",
            actions_taken=["read_logs", "read_metrics", "read_logs", "identify_cause", "rollback", "monitor_recovery", "resolve"],
            rollback_count=1,
            escalation_count=0,
            already_read_logs={"api-gateway", "order-service"},
            already_read_metrics={"api-gateway"},
            steps_used=7,
        )
        assert clamped > 0
        assert bd["service_recovery"] == 30.0
        assert bd["root_cause_accuracy"] == 25.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
