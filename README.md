---
title: Incident Commander Environment Server
emoji: 🎣
colorFrom: purple
colorTo: red
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Incident Commander Environment

An OpenEnv RL environment where an AI agent investigates and resolves microservice outages. Theme 3.1 compliant: real stateful tool interactions, partial observability, causal chain scenarios, multi-round fix loops, and 4 independent reward signals.

## Quick Start

```python
from incident_commander import IncidentCommanderEnv, IncidentCommanderAction

# Connect to the deployed Space or local server
env = IncidentCommanderEnv(base_url="http://localhost:8000")
obs = env.reset()
print(obs.alert_summary)   # e.g. "[CRITICAL] PagerDuty: payment-service error rate critical"

# Investigate
obs = env.step(IncidentCommanderAction(
    action_type="read_logs",
    target_service="payment-service",
))
print(obs.revealed_logs)

# Lock hypothesis
obs = env.step(IncidentCommanderAction(
    action_type="identify_cause",
    target_service="payment-service",
    hypothesis="memory_limit_too_low",
))

# Apply fix
obs = env.step(IncidentCommanderAction(
    action_type="scale_up",
    target_service="payment-service",
))

# Monitor and resolve
obs = env.step(IncidentCommanderAction(action_type="monitor_recovery"))
obs = env.step(IncidentCommanderAction(action_type="resolve", justification="Service recovered"))

env.close()
```

## Root Cause → Fix Reference

| Root Cause | Correct Fix |
|------------|-------------|
| `memory_limit_too_low` | `scale_up` |
| `bad_deployment` | `rollback` |
| `connection_pool_exhausted` | `hotfix` |
| `traffic_spike` | `scale_up` |
| `dependency_failure` | `restart_pod` |
| `config_error` | `hotfix` |
| `redis_down` | `restart_pod` |
| `certificate_expired` | `hotfix` |

## Advanced Usage

### Connecting to an Existing Server

If you already have a Incident Commander environment server running, you can connect directly:

```python
from incident_commander import IncidentCommanderEnv, IncidentCommanderAction

# Connect to existing server
env = IncidentCommanderEnv(base_url="http://localhost:8000")

# Use as normal
obs = env.reset()
obs = env.step(IncidentCommanderAction(
    action_type="read_logs",
    target_service="payment-service",
))
```

Note: When connecting to an existing server, `env.close()` will NOT stop the server.

### Using the Context Manager

The client supports context manager usage for automatic connection management:

```python
from incident_commander import IncidentCommanderEnv, IncidentCommanderAction

with IncidentCommanderEnv(base_url="http://localhost:8000") as env:
    obs = env.reset()
    for svc in ["payment-service", "auth-service"]:
        obs = env.step(IncidentCommanderAction(
            action_type="read_logs",
            target_service=svc,
        ))
        print(f"Logs for {svc}: {obs.revealed_logs.get(svc)}")
```

The client uses WebSocket connections for:
- **Lower latency**: No HTTP connection overhead per request
- **Persistent session**: Server maintains your environment state
- **Efficient for episodes**: Better for many sequential steps

### Concurrent WebSocket Sessions

The server supports up to 16 concurrent WebSocket connections (configured in `server/app.py`):

```python
from incident_commander import IncidentCommanderEnv, IncidentCommanderAction
from concurrent.futures import ThreadPoolExecutor

def run_episode(client_id: int):
    with IncidentCommanderEnv(base_url="http://localhost:8000") as env:
        env.reset()
        for i in range(10):
            obs = env.step(IncidentCommanderAction(
                action_type="read_logs",
                target_service="payment-service",
            ))
        return client_id, obs.reward

with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(run_episode, range(4)))
```

## Development & Testing

### Direct Environment Testing

Test the environment logic directly without starting the HTTP server:

```bash
# From the server directory
python3 server/incident_commander_environment.py
```

This verifies that:
- Environment resets correctly
- Step executes actions properly
- State tracking works
- Rewards are calculated correctly

### Running Locally

Run the server locally for development:

```bash
uvicorn server.app:app --reload
```

## Project Structure

```
incident_commander/
├── .dockerignore         # Docker build exclusions
├── __init__.py            # Module exports
├── README.md            # This file
├── DEPLOY.md            # HuggingFace + Colab deployment guide
├── openenv.yaml         # OpenEnv manifest
├── pyproject.toml       # Project metadata and dependencies
├── colab/
│   └── incident_commander_training.ipynb  # Colab training notebook
├── client.py            # IncidentCommanderEnv client
├── models.py            # Action and Observation models
├── rewards.py           # 4-signal reward functions
├── simulator.py        # Scenario bank + log/metric generators
├── training.py         # GRPO + Unsloth training script
├── inference.py        # Trained model evaluation
└── server/
    ├── __init__.py        # Server module exports
    ├── incident_commander_environment.py  # Core environment logic
    ├── app.py             # FastAPI application (HTTP + WebSocket endpoints)
    └── Dockerfile        # Container image definition

## Phase 3: Model Training (TRL GRPO + Unsloth)

We've set up an RL training pipeline explicitly designed for Theme 3.1 compliance. Using Unsloth for ultra-fast generation and TRL's internal GRPO algorithm, this pipeline uses proximal policy optimization over multiple environment steps.

### Training

To execute the training loop (requires Unsloth, TRL, pyarrow etc.):
```bash
python training.py --run
```

This starts by loading `unsloth/Llama-3.2-1B-Instruct` and fine-tunes it natively against the `IncidentCommanderEnvironment`. The trainer simulates `identify_cause`, `read_logs`, etc., directly extracting the reward computed by our 4 anti-shortcut reward modules.

The trained model saves to `outputs/commander_final`.

### Inference Evaluation

To evaluate the trained model iteratively interacting with an episode inside the environment:
```bash
python inference.py --model outputs/commander_final --difficulty 1
```

The script runs the model through a 50-step max causal scenario until it correctly issues a resolve action or runs out of time.

---

## Deployment

### Hugging Face Spaces

Deploy the environment as a shareable Hugging Face Space:

```bash
# Build Docker image
docker build -t incident_commander-env:latest -f server/Dockerfile .

# Push to Hugging Face Spaces
openenv push
```

See [DEPLOY.md](DEPLOY.md) for full instructions including client connection examples.

### Google Colab Training

Open `colab/incident_commander_training.ipynb` in Colab for a complete training notebook with:

- One-click dependency installation
- Unsloth 4-bit quantized model loading
- GRPO trainer with dual reward functions
- Evaluation episode runner
- Model download / push to Hugging Face Hub

**Quick start in Colab:**
```bash
!pip install -q openenv-core[core] unsloth trl datasets peft accelerate bitsandbytes scipy
!python training.py --run
```
