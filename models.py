# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Incident Commander Environment.

The incident commander environment trains agents to investigate and resolve
microservice outages using real tool interactions and causal reasoning.
"""

from typing import Dict, List, Literal, Optional, Any
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
        None, description="The service this action targets (optional for some actions, like resolve)."
    )
    hypothesis: Optional[str] = Field(
        None,
        description=(
            "Root cause hypothesis (required for identify_cause). "
            "Must be one of: memory_limit_too_low, bad_deployment, "
            "connection_pool_exhausted, traffic_spike, dependency_failure, "
            "config_error, redis_down, certificate_expired."
        ),
    )
    justification: Optional[str] = Field(
        None, description="Justification text (required for escalate/resolve)."
    )
    time_range_minutes: Optional[int] = Field(
        5,
        description=(
            "How many minutes of historical data to request for logs/metrics. "
            "Larger ranges cost more steps but reveal more."
        ),
    )


class RewardBreakdown(Observation):
    """Internal reward breakdown — returned so training can log each signal."""

    service_recovery: float = Field(default=0.0)
    root_cause_accuracy: float = Field(default=0.0)
    action_quality: float = Field(default=0.0)
    speed: float = Field(default=0.0)
    total: float = Field(default=0.0)


class IncidentCommanderObservation(Observation):
    """Observation from the Incident Commander environment."""

    # Partial observability — only what the agent has explicitly revealed
    alert_summary: str = Field(
        ..., description="Active PagerDuty-style alert headline."
    )
    services_overview: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="High-level status of all services (name, status, error_rate).",
    )
    revealed_logs: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Logs revealed so far (service → list of log lines). Only includes services the agent has read.",
    )
    revealed_metrics: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Metrics revealed so far (service → metric dict). Only includes services the agent has read.",
    )
    deployment_history: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Recent deployment history — only available after read_deployment_history.",
    )
    dependency_graph: Optional[Dict[str, List[str]]] = Field(
        default=None,
        description="Service dependency graph — only available after read_dependency_graph.",
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
        default=None, description="The committed hypothesis value, if locked."
    )
    post_fix_status: Optional[str] = Field(
        default=None,
        description=(
            "Service status observed AFTER applying a fix: "
            "'recovered' | 'degraded' | 'worse' | None (no fix applied yet)."
        ),
    )
    steps_remaining: int = Field(
        default=50, description="Steps remaining before timeout."
    )
    last_action_result: str = Field(
        default="", description="Human-readable result of the last action."
    )
    reward_breakdown: Optional[Dict[str, float]] = Field(
        default=None,
        description="Per-signal reward breakdown (only populated on done=True).",
    )
