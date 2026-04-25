"""
FastAPI application for the Incident Commander Environment.

Exposes the IncidentCommanderEnvironment over HTTP and WebSocket endpoints
compatible with EnvClient. Mounts a Gradio /web UI when gradio is installed.
"""

import json

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv-core is required. Install via `pip install openenv-core[core]`."
    ) from e

# Resilient imports - work whether server.app is run as a module from the
# repo root, as a top-level script, or as part of the installed
# incident_commander package.
try:
    from server.incident_commander_environment import IncidentCommanderEnvironment
    from models import IncidentCommanderAction, IncidentCommanderObservation
except (ImportError, ModuleNotFoundError):
    try:
        from incident_commander_environment import IncidentCommanderEnvironment  # type: ignore
        from models import IncidentCommanderAction, IncidentCommanderObservation
    except (ImportError, ModuleNotFoundError):
        from incident_commander.server.incident_commander_environment import (
            IncidentCommanderEnvironment,
        )
        from incident_commander.models import (
            IncidentCommanderAction, IncidentCommanderObservation,
        )

try:
    import gradio as gr
except Exception:  # pragma: no cover
    gr = None


# Create the OpenEnv FastAPI app with HTTP + WS endpoints + /docs + /health.
app = create_app(
    IncidentCommanderEnvironment,
    IncidentCommanderAction,
    IncidentCommanderObservation,
    env_name="incident_commander",
    max_concurrent_envs=16,
)


def _build_gradio_demo():
    """Lightweight Gradio UI for manual reset/step testing."""
    env = IncidentCommanderEnvironment()

    def _to_json(value):
        if hasattr(value, "model_dump"):
            payload = value.model_dump()
        elif hasattr(value, "dict"):
            payload = value.dict()
        else:
            payload = value
        return json.dumps(payload, indent=2, ensure_ascii=False, default=str)

    def do_reset(difficulty):
        obs = env.reset(difficulty=difficulty)
        return _to_json(obs)

    def do_step(action_type, target_service, hypothesis, justification, time_range_minutes):
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
        "read_logs", "read_metrics",
        "read_deployment_history", "read_dependency_graph",
        "identify_cause",
        "restart_pod", "rollback", "scale_up", "hotfix",
        "escalate", "monitor_recovery", "resolve",
    ]

    with gr.Blocks(title="Incident Commander tester") as demo:
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
            inputs=[action_type, target_service, hypothesis,
                    justification, time_range_minutes],
            outputs=[output],
        )
    return demo


if gr is not None:
    try:
        app = gr.mount_gradio_app(app, _build_gradio_demo(), path="/web")
    except Exception as e:  # pragma: no cover
        # Gradio mounting is best-effort; the API endpoints still work.
        print(f"[warn] could not mount Gradio /web demo: {e}")


def main(host="0.0.0.0", port=8000):
    """CLI entrypoint registered as `server` in pyproject.toml."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(host=args.host, port=args.port)
