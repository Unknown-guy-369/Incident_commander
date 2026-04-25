"""
Simulator: scenario bank and dynamic log/metric generators.

All scenarios are built around causal chains. The same root cause
produces randomized surface symptoms each episode so agents cannot
memorize symptom→fix mappings — they must actually read and reason.
"""

from __future__ import annotations

import random
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Root Cause Registry
# ---------------------------------------------------------------------------

VALID_ROOT_CAUSES = [
    "memory_limit_too_low",
    "bad_deployment",
    "connection_pool_exhausted",
    "traffic_spike",
    "dependency_failure",
    "config_error",
    "redis_down",
    "certificate_expired",
]

# Maps root cause → correct fix action
CORRECT_FIX = {
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
# Scenario Dataclass
# ---------------------------------------------------------------------------

@dataclass
class Scenario:
    name: str
    root_cause: str
    correct_fix: str

    # Services involved in the causal chain
    root_service: str           # where the root cause lives
    affected_services: List[str]  # downstream victims
    all_services: List[str]     # full service mesh

    # Templates used by generators
    _error_log_templates: List[str] = field(default_factory=list)
    _red_herring_services: Dict[str, str] = field(default_factory=dict)
    _deployment_history: List[Dict[str, Any]] = field(default_factory=list)
    _dependency_graph: Dict[str, List[str]] = field(default_factory=dict)

    # Randomised per-episode (filled by Simulator.sample)
    seed: int = 0


# ---------------------------------------------------------------------------
# Scenario Bank
# ---------------------------------------------------------------------------

ALL_SCENARIOS: List[Scenario] = [

    # ── Scenario 1: OOM Kill (Level 1) ──────────────────────────────────
    Scenario(
        name="OOM Kill — payment-service",
        root_cause="memory_limit_too_low",
        correct_fix="scale_up",
        root_service="payment-service",
        affected_services=["payment-service"],
        all_services=["api-gateway", "payment-service", "auth-service", "order-service"],
        _error_log_templates=[
            "FATAL payment-service: OOMKilled — container exceeded memory limit (256Mi)",
            "WARNING payment-service: memory usage at {mem}% — approaching limit",
            "ERROR payment-service: request failed: 500 Internal Server Error",
        ],
        _red_herring_services={"auth-service": "CPU spike (routine batch job)"},
        _deployment_history=[
            {"service": "payment-service", "version": "v1.4.2", "time": "-3h", "status": "ok"},
            {"service": "api-gateway", "version": "v3.0.1", "time": "-1d", "status": "ok"},
        ],
        _dependency_graph={
            "api-gateway":   ["payment-service", "auth-service"],
            "payment-service": [],
            "auth-service":  [],
            "order-service": ["payment-service"],
        },
    ),

    # ── Scenario 2: Bad Deploy (Level 1) ────────────────────────────────
    Scenario(
        name="Bad Deploy — api-gateway NullPointerException",
        root_cause="bad_deployment",
        correct_fix="rollback",
        root_service="api-gateway",
        affected_services=["api-gateway"],
        all_services=["api-gateway", "payment-service", "auth-service", "order-service"],
        _error_log_templates=[
            "ERROR api-gateway: java.lang.NullPointerException at RequestHandler.java:142",
            "ERROR api-gateway: 500 responses spiking — {errors}/min errors",
            "INFO api-gateway: deployment v2.1.3 rolled out at {deploy_time}",
        ],
        _red_herring_services={"order-service": "slightly elevated DB latency (unrelated)"},
        _deployment_history=[
            {"service": "api-gateway",  "version": "v2.1.3", "time": "-12min", "status": "suspect"},
            {"service": "api-gateway",  "version": "v2.1.2", "time": "-2d",    "status": "ok"},
            {"service": "order-service","version": "v1.0.9", "time": "-1d",    "status": "ok"},
        ],
        _dependency_graph={
            "api-gateway":   ["payment-service", "auth-service", "order-service"],
            "payment-service": [],
            "auth-service":  [],
            "order-service": ["payment-service"],
        },
    ),

    # ── Scenario 3: DB Connection Pool Exhausted (Level 2 cascade) ──────
    Scenario(
        name="DB Connection Pool Exhausted — order + inventory cascade",
        root_cause="connection_pool_exhausted",
        correct_fix="hotfix",
        root_service="db-proxy",
        affected_services=["order-service", "inventory-service"],
        all_services=["api-gateway", "order-service", "inventory-service", "db-proxy", "auth-service"],
        _error_log_templates=[
            "ERROR order-service: too many connections — pool exhausted ({pool}/100)",
            "ERROR inventory-service: connection timeout waiting for DB slot",
            "WARN db-proxy: connection pool at {pool}% capacity",
        ],
        _red_herring_services={"auth-service": "timeout errors — secondary effect of order-service slowness"},
        _deployment_history=[
            {"service": "order-service", "version": "v2.3.0", "time": "-5d", "status": "ok"},
        ],
        _dependency_graph={
            "api-gateway":      ["order-service", "inventory-service", "auth-service"],
            "order-service":    ["db-proxy"],
            "inventory-service":["db-proxy"],
            "db-proxy":         [],
            "auth-service":     [],
        },
    ),

    # ── Scenario 4: Redis Down → Auth slow → API timeouts (Causal chain) ─
    Scenario(
        name="Redis Cache Down — multi-hop causal chain",
        root_cause="redis_down",
        correct_fix="restart_pod",
        root_service="redis-cache",
        affected_services=["auth-service", "api-gateway"],
        all_services=["api-gateway", "auth-service", "redis-cache", "payment-service", "order-service"],
        _error_log_templates=[
            "ERROR redis-cache: connection refused — pod not responding",
            "WARN auth-service: cache miss rate 100% — Redis unreachable, response time +{latency}ms",
            "ERROR api-gateway: timeout waiting for auth-service — {timeout_count} timeouts/min",
        ],
        _red_herring_services={"payment-service": "elevated error rate (downstream of api-gateway)"},
        _deployment_history=[
            {"service": "redis-cache", "version": "v6.2.1", "time": "-2d", "status": "ok"},
        ],
        _dependency_graph={
            "api-gateway":   ["auth-service", "payment-service", "order-service"],
            "auth-service":  ["redis-cache"],
            "redis-cache":   [],
            "payment-service":[],
            "order-service": [],
        },
    ),

    # ── Scenario 5: Time-delayed memory leak (Causal reasoning) ─────────
    Scenario(
        name="Memory Leak — order-service v2.3.1 deployed 2h ago",
        root_cause="bad_deployment",
        correct_fix="rollback",
        root_service="order-service",
        affected_services=["order-service"],
        all_services=["api-gateway", "order-service", "inventory-service", "auth-service"],
        _error_log_templates=[
            "WARN order-service: memory usage climbing — {mem}% (deployed v2.3.1 2h ago)",
            "ERROR order-service: OOMKilled — memory leak suspected in request handler",
            "INFO order-service: heap dump analysis shows unclosed streams in v2.3.1",
        ],
        _red_herring_services={"inventory-service": "traffic spike (normal flash sale traffic)"},
        _deployment_history=[
            {"service": "order-service", "version": "v2.3.1", "time": "-2h",  "status": "suspect"},
            {"service": "order-service", "version": "v2.3.0", "time": "-5d", "status": "ok"},
        ],
        _dependency_graph={
            "api-gateway":      ["order-service", "auth-service"],
            "order-service":    ["inventory-service"],
            "inventory-service":[],
            "auth-service":     [],
        },
    ),

    # ── Scenario 6: Traffic Spike (Level 2) ─────────────────────────────
    Scenario(
        name="Traffic Spike — payment-service under load",
        root_cause="traffic_spike",
        correct_fix="scale_up",
        root_service="payment-service",
        affected_services=["payment-service"],
        all_services=["api-gateway", "payment-service", "auth-service", "order-service"],
        _error_log_templates=[
            "WARN payment-service: RPS jumped to {rps} — pod count insufficient",
            "ERROR payment-service: response time at {latency}ms — SLA breach",
            "WARN api-gateway: upstream payment-service slow — circuit breaker threshold approaching",
        ],
        _red_herring_services={"auth-service": "normal CPU increase (correlated but not causal)"},
        _deployment_history=[
            {"service": "payment-service", "version": "v1.4.2", "time": "-1d", "status": "ok"},
        ],
        _dependency_graph={
            "api-gateway":   ["payment-service", "auth-service", "order-service"],
            "payment-service": [],
            "auth-service":  [],
            "order-service": ["payment-service"],
        },
    ),

    # ── Scenario 7: Config Error ─────────────────────────────────────────
    Scenario(
        name="Config Error — invalid environment variable",
        root_cause="config_error",
        correct_fix="hotfix",
        root_service="inventory-service",
        affected_services=["inventory-service"],
        all_services=["api-gateway", "order-service", "inventory-service", "auth-service"],
        _error_log_templates=[
            "ERROR inventory-service: failed to parse DATABASE_URL — invalid config",
            "FATAL inventory-service: startup failed — missing required env var WAREHOUSE_API_KEY",
            "ERROR api-gateway: inventory-service unavailable — 503",
        ],
        _red_herring_services={"order-service": "request queue building up (downstream effect)"},
        _deployment_history=[
            {"service": "inventory-service", "version": "v1.2.0", "time": "-30min", "status": "suspect"},
        ],
        _dependency_graph={
            "api-gateway":      ["order-service", "inventory-service"],
            "order-service":    ["inventory-service"],
            "inventory-service":[],
            "auth-service":     [],
        },
    ),

    # ── Scenario 8: Expired Certificate ─────────────────────────────────
    Scenario(
        name="Expired TLS Certificate — auth-service HTTPS failure",
        root_cause="certificate_expired",
        correct_fix="hotfix",
        root_service="auth-service",
        affected_services=["auth-service", "api-gateway"],
        all_services=["api-gateway", "auth-service", "payment-service", "order-service"],
        _error_log_templates=[
            "ERROR auth-service: TLS handshake failed — certificate expired {days_ago} day(s) ago",
            "ERROR api-gateway: SSL CERTIFICATE VERIFY FAILED connecting to auth-service",
            "WARN payment-service: auth validation errors — 401 responses increasing",
        ],
        _red_herring_services={"order-service": "order failures (downstream of auth breakage)"},
        _deployment_history=[
            {"service": "auth-service", "version": "v2.0.5", "time": "-3d", "status": "ok"},
        ],
        _dependency_graph={
            "api-gateway":   ["auth-service", "payment-service", "order-service"],
            "auth-service":  [],
            "payment-service":[],
            "order-service": ["auth-service"],
        },
    ),
]


# ---------------------------------------------------------------------------
# Metric Generator
# ---------------------------------------------------------------------------

def _base_metrics(service: str, is_affected: bool, rng: random.Random) -> Dict[str, Any]:
    """Generate dynamic per-service metrics. Values are randomised so the
    agent cannot memorise numbers but patterns remain consistent."""
    if is_affected:
        return {
            "cpu_pct":    rng.randint(60, 95),
            "memory_pct": rng.randint(85, 99),
            "error_rate": rng.randint(40, 95),
            "p99_ms":     rng.randint(3000, 9000),
            "rps":        rng.randint(200, 800),
        }
    else:
        return {
            "cpu_pct":    rng.randint(10, 35),
            "memory_pct": rng.randint(20, 50),
            "error_rate": rng.randint(0, 3),
            "p99_ms":     rng.randint(50, 200),
            "rps":        rng.randint(50, 300),
        }


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

class Simulator:
    """Draws a random scenario each episode and generates dynamic observations."""

    _global_counter: int = 0

    def __init__(self, difficulty: int = 1):
        """
        difficulty: 1 (easy, 1-service) → 4 (full, multi-hop causal chains)
        """
        self.difficulty = difficulty
        self._rng = random.Random()

    def sample(self) -> "EpisodeContext":
        """Sample a scenario and seed for this episode."""
        Simulator._global_counter += 1
        seed = (int(time.time() * 1000) + Simulator._global_counter * 7919) % (2**31)
        rng = random.Random(seed)

        # Filter by difficulty
        if self.difficulty == 1:
            pool = [s for s in ALL_SCENARIOS if len(s.affected_services) == 1
                    and "cascade" not in s.name.lower()
                    and "causal" not in s.name.lower()]
        elif self.difficulty == 2:
            pool = [s for s in ALL_SCENARIOS if len(s.affected_services) <= 2]
        else:
            pool = ALL_SCENARIOS

        scenario = deepcopy(rng.choice(pool))
        scenario.seed = seed
        return EpisodeContext(scenario=scenario, rng=rng)


class EpisodeContext:
    """Holds all dynamic state for a single episode."""

    def __init__(self, scenario: Scenario, rng: random.Random):
        self.scenario = scenario
        self._rng = rng
        self._revealed_logs: Dict[str, List[str]] = {}
        self._revealed_metrics: Dict[str, Dict[str, Any]] = {}
        self._deployment_history_revealed = False
        self._dependency_graph_revealed = False

        # Generate full (hidden) metrics for all services
        self._full_metrics: Dict[str, Dict[str, Any]] = {
            svc: _base_metrics(svc, svc in scenario.affected_services or svc == scenario.root_service, rng)
            for svc in scenario.all_services
        }

        # Inject scenario-specific overrides into metrics
        self._inject_scenario_metrics()

    def _inject_scenario_metrics(self):
        sc = self.scenario
        rng = self._rng
        if sc.root_cause == "memory_limit_too_low":
            if sc.root_service in self._full_metrics:
                self._full_metrics[sc.root_service]["memory_pct"] = rng.randint(97, 99)
                self._full_metrics[sc.root_service]["oom_kills"] = rng.randint(2, 8)
        elif sc.root_cause == "connection_pool_exhausted":
            for svc in sc.affected_services:
                if svc in self._full_metrics:
                    self._full_metrics[svc]["db_pool_used"] = 100
                    self._full_metrics[svc]["db_pool_total"] = 100
        elif sc.root_cause == "traffic_spike":
            if sc.root_service in self._full_metrics:
                self._full_metrics[sc.root_service]["rps"] = rng.randint(1200, 3000)
                self._full_metrics[sc.root_service]["pod_count"] = 2  # not enough pods

    # ── Log revelation ────────────────────────────────────────────────────

    def reveal_logs(self, service: str, time_range_minutes: int) -> List[str]:
        """Return log lines for `service`. Only red-herring content for
        services that are not directly affected."""
        if service in self._revealed_logs:
            return self._revealed_logs[service]  # already read — no new info

        rng = self._rng
        sc = self.scenario
        lines: List[str] = []

        is_root = (service == sc.root_service)
        is_affected = service in sc.affected_services

        if is_root or is_affected:
            # Pick 2-4 symptom lines from templates, interpolate values
            templates = sc._error_log_templates
            n = min(len(templates), rng.randint(2, 4))
            chosen = rng.sample(templates, n)
            for tpl in chosen:
                line = tpl.format(
                    mem=rng.randint(90, 99),
                    errors=rng.randint(50, 200),
                    deploy_time=f"14:{rng.randint(10,59):02d}",
                    pool=rng.randint(90, 100),
                    latency=rng.randint(400, 1500),
                    timeout_count=rng.randint(20, 80),
                    rps=rng.randint(1200, 3000),
                    days_ago=rng.randint(1, 5),
                )
                lines.append(line)
        elif service in sc._red_herring_services:
            lines.append(f"INFO {service}: {sc._red_herring_services[service]}")
        else:
            lines.append(f"INFO {service}: all systems normal")

        # Add time-range noise — longer range reveals older (irrelevant) lines
        if time_range_minutes > 15:
            lines.append(f"DEBUG {service}: scheduled GC ran, paused {rng.randint(10,200)}ms ago")

        self._revealed_logs[service] = lines
        return lines

    def reveal_metrics(self, service: str) -> Dict[str, Any]:
        if service in self._revealed_metrics:
            return self._revealed_metrics[service]
        metrics = self._full_metrics.get(service, {"error": "service not found"})
        self._revealed_metrics[service] = metrics
        return metrics

    def reveal_deployment_history(self) -> List[Dict[str, Any]]:
        self._deployment_history_revealed = True
        return self.scenario._deployment_history

    def reveal_dependency_graph(self) -> Dict[str, List[str]]:
        self._dependency_graph_revealed = True
        return self.scenario._dependency_graph

    # ── Services overview (always visible) ───────────────────────────────

    def services_overview(self) -> list:
        result = []
        for svc in self.scenario.all_services:
            m = self._full_metrics[svc]
            result.append({
                "name":        svc,
                "status":      "degraded" if m["error_rate"] > 20 else "healthy",
                "error_rate":  m["error_rate"],
            })
        return result

    # ── Alert ─────────────────────────────────────────────────────────────

    def alert_summary(self) -> str:
        sc = self.scenario
        worst = sc.affected_services[0] if sc.affected_services else sc.root_service
        return (
            f"[CRITICAL] PagerDuty: {worst} error rate critical — "
            f"incident #{self._rng.randint(10000, 99999)} opened"
        )
