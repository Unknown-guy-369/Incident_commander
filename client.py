"""
Client for the Incident Commander Environment.
"""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import IncidentCommanderAction, IncidentCommanderObservation


class IncidentCommanderEnv(
    EnvClient[IncidentCommanderAction, IncidentCommanderObservation, State]
):
    """
    Client for the Incident Commander Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.

    Example:
        >>> with IncidentCommanderEnv(base_url="http://localhost:8000") as env:
        ...     result = env.reset()
        ...     print(result.observation.alert_summary)
        ...     result = env.step(IncidentCommanderAction(
        ...         action_type="read_logs",
        ...         target_service="payment-service",
        ...     ))
    """

    def _step_payload(self, action: IncidentCommanderAction) -> Dict:
        payload = {
            "action_type": action.action_type,
            "target_service": action.target_service,
        }
        if action.hypothesis is not None:
            payload["hypothesis"] = action.hypothesis
        if action.justification is not None:
            payload["justification"] = action.justification
        if action.time_range_minutes is not None:
            payload["time_range_minutes"] = action.time_range_minutes
        return payload

    def _parse_result(self, payload: Dict) -> StepResult[IncidentCommanderObservation]:
        obs_data = payload.get("observation", {})
        observation = IncidentCommanderObservation(
            alert_summary=obs_data.get("alert_summary", ""),
            services_overview=obs_data.get("services_overview", []),
            revealed_logs=obs_data.get("revealed_logs", {}),
            revealed_metrics=obs_data.get("revealed_metrics", {}),
            deployment_history=obs_data.get("deployment_history"),
            dependency_graph=obs_data.get("dependency_graph"),
            actions_taken=obs_data.get("actions_taken", []),
            hypothesis_locked=obs_data.get("hypothesis_locked", False),
            locked_hypothesis=obs_data.get("locked_hypothesis"),
            post_fix_status=obs_data.get("post_fix_status"),
            steps_remaining=obs_data.get("steps_remaining", 50),
            last_action_result=obs_data.get("last_action_result", ""),
            reward_breakdown=obs_data.get("reward_breakdown"),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id", ""),
            step_count=payload.get("step_count", 0),
        )
