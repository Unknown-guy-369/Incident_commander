"""
Data models for the Incident Commander Environment.
"""

from typing import Any, Dict, List, Literal, Optional
from pydantic import Field
from openenv.core.env_server.types import Action, Observation


ActionType = Literal[
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
]


class IncidentCommanderAction(Action):
    """Action for the Incident Commander environment."""

    action_type: ActionType = Field(
        ..., description="Type of action to perform."
    )
    target_service: Optional[str] = Field(
        None, description="The service this action targets (optional for some actions)."
    )
    hypothesis: Optional[str] = Field(
        None,
        description=(
            "Root cause hypothesis (required for identify_cause). One of: "
            "memory_limit_too_low, bad_deployment, connection_pool_exhausted, "
            "traffic_spike, dependency_failure, config_error, redis_down, "
            "certificate_expired."
        ),
    )
    justification: Optional[str] = Field(
        None, description="Justification text (used by escalate / resolve)."
    )
    time_range_minutes: Optional[int] = Field(
        5, description="Minutes of historical data to request for logs/metrics."
    )


class IncidentCommanderObservation(Observation):
    """Observation from the Incident Commander environment."""

    alert_summary: str = Field(
        ..., description="Active PagerDuty-style alert headline."
    )
    services_overview: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="High-level status of all services (name, status, error_rate).",
    )
    revealed_logs: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Service -> revealed log lines so far.",
    )
    revealed_metrics: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Service -> revealed metrics dict so far.",
    )
    deployment_history: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Recent deployment history (only after read_deployment_history).",
    )
    dependency_graph: Optional[Dict[str, List[str]]] = Field(
        default=None,
        description="Service dependency graph (only after read_dependency_graph).",
    )
    actions_taken: List[str] = Field(
        default_factory=list,
        description="History of action_type values taken so far in this episode.",
    )
    hypothesis_locked: bool = Field(
        default=False,
        description="Whether the agent has committed to a root cause hypothesis.",
    )
    locked_hypothesis: Optional[str] = Field(
        default=None, description="Committed hypothesis value, if locked."
    )
    post_fix_status: Optional[str] = Field(
        default=None,
        description=(
            "Service status AFTER applying a fix: "
            "'recovered' | 'degraded' | 'worse' | None."
        ),
    )
    steps_remaining: int = Field(
        default=50, description="Steps remaining before timeout."
    )
    last_action_result: str = Field(
        default="", description="Human-readable result of the last action."
    )
    reward_breakdown: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Per-signal reward breakdown (populated only on done=True). "
            "Holds the four numeric rewards plus terminal metadata such as "
            "true_root_cause and root_service."
        ),
    )
