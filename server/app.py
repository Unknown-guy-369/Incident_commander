# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Incident Commander Environment.

This module creates an HTTP server that exposes the IncidentCommanderEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions
    - GET /gradio: Minimal Gradio interface for reset/step testing

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

import json

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from models import IncidentCommanderAction, IncidentCommanderObservation
    from incident_commander_environment import IncidentCommanderEnvironment
except ModuleNotFoundError:
    from models import IncidentCommanderAction, IncidentCommanderObservation
    from server.incident_commander_environment import IncidentCommanderEnvironment

try:
    import gradio as gr
except Exception:  # pragma: no cover
    gr = None


# Create the app with web interface and README integration
app = create_app(
    IncidentCommanderEnvironment,
    IncidentCommanderAction,
    IncidentCommanderObservation,
    env_name="incident_commander",
    max_concurrent_envs=16,  # Up to 16 parallel GRPO rollout workers
)


def _build_gradio_demo():
    """Create a lightweight Gradio app for manual reset/step testing."""
    env = IncidentCommanderEnvironment()

    def _to_json(value):
        if hasattr(value, "model_dump"):
            payload = value.model_dump()
        elif hasattr(value, "dict"):
            payload = value.dict()
        else:
            payload = value
        return json.dumps(payload, indent=2, ensure_ascii=False, default=str)

    def do_reset(difficulty: int):
        obs = env.reset(difficulty=difficulty)
        return _to_json(obs)

    def do_step(
        action_type: str,
        target_service: str,
        hypothesis: str,
        justification: str,
        time_range_minutes: int,
    ):
        action = IncidentCommanderAction(
            action_type=action_type,
            target_service=target_service.strip() or "payment-service",
            hypothesis=hypothesis.strip() or None,
            justification=justification.strip() or None,
            time_range_minutes=time_range_minutes,
        )
        obs = env.step(action)
        return _to_json(obs)

    action_choices = [
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

    with gr.Blocks(title="Incident Commander Step/Reset") as demo:
        gr.Markdown(
            "## Incident Commander tester\n"
            "Use **Reset** to start a new incident and **Step** to execute actions."
        )
        with gr.Row():
            difficulty = gr.Slider(1, 4, value=1, step=1, label="Difficulty")
            reset_btn = gr.Button("Reset", variant="primary")
        with gr.Row():
            action_type = gr.Dropdown(
                choices=action_choices, value="read_logs", label="Action Type"
            )
            target_service = gr.Textbox(value="payment-service", label="Target Service")
        with gr.Row():
            hypothesis = gr.Textbox(label="Hypothesis (identify_cause)")
            justification = gr.Textbox(label="Justification (escalate/resolve)")
            time_range_minutes = gr.Slider(
                1, 60, value=5, step=1, label="Time Range Minutes"
            )
        step_btn = gr.Button("Step")
        output = gr.Code(
            label="Observation JSON",
            language="json",
            value='{"message": "Press Reset to begin"}',
        )
        reset_btn.click(do_reset, inputs=[difficulty], outputs=[output])
        step_btn.click(
            do_step,
            inputs=[
                action_type,
                target_service,
                hypothesis,
                justification,
                time_range_minutes,
            ],
            outputs=[output],
        )
    return demo


if gr is not None:
    app = gr.mount_gradio_app(app, _build_gradio_demo(), path="/gradio")


def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m incident_commander.server.app

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 8000)

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn incident_commander.server.app:app --workers 4
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
