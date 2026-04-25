# Incident Commander — Deployment Guide

This guide covers deploying the environment to **Hugging Face Spaces** and running training in **Google Colab**.

---

## Part 1: Hugging Face Spaces Deployment

### Prerequisites

- Docker installed locally
- Hugging Face account (free tier works)
- `openenv` CLI installed: `pip install openenv`
- Login to Hugging Face: `huggingface-cli login`

### Step 1 — Build the Docker Image

From the project root:

```bash
docker build -t incident_commander-env:latest -f server/Dockerfile .
```

### Step 2 — Push to Hugging Face Spaces

```bash
# From the environment directory
cd incident_commander

# Push (follow the prompts)
openenv push

# Or with options
openenv push --namespace your-username --repo-id your-username/incident-commander
```

The `openenv push` command will:
1. Validate the directory structure
2. Build a Hugging Face-optimized Docker image
3. Upload to your Space
4. Provide the deployment URL

### Step 3 — Verify Deployment

After push completes, your Space will be live at:
```
https://huggingface.co/spaces/<namespace>/incident-commander
```

Endpoints available:
- `/web` — Interactive web UI for manual testing
- `/docs` — OpenAPI/Swagger documentation
- `/health` — Health check
- `/ws` — WebSocket for persistent sessions

### Step 4 — Connect a Client

```python
# Option A: Direct HTTP
from incident_commander import IncidentCommanderEnv, IncidentCommanderAction

env = IncidentCommanderEnv(base_url="https://your-space.huggingface.co")
result = env.reset()
print(result.observation.alert_summary)

result = env.step(IncidentCommanderAction(
    action_type="read_logs",
    target_service="payment-service",
))
print(result.observation.revealed_logs)

env.close()

# Option B: Context manager
with IncidentCommanderEnv(base_url="https://your-space.huggingface.co") as env:
    result = env.reset()
    ...
```

### Sharing the Space

Once deployed, share these with your team:
1. The Space URL
2. The action schema (from `/docs`)
3. Example usage snippets

---

## Part 2: Training in Google Colab

Open the notebook in Colab: `colab/incident_commander_training.ipynb`

### Quick Start

```bash
# Step 1: Install dependencies
!pip install openenv-core[core] unsloth trl datasets peft accelerate bitsandbytes

# Step 2: Clone / upload the project
# If using from GitHub:
!git clone https://github.com/your-username/incident-commander.git
%cd incident-commander

# Step 3: Run training
!python training.py --run
```

### Expected Training Output

```
Loading lightweight model via Unsloth...
Generating curriculum tasks...
Initializing GRPOTrainer...
Starting Phase 3 Training (GRPO + Unsloth)...
  Step | Loss | Reward | Time
  10   | 0.42 | 0.31  | 12s
  20   | 0.38 | 0.45  | 24s
  ...

Saving trained model...
Model saved to outputs/commander_final
```

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MODEL_NAME` | `unsloth/Llama-3.2-1B-Instruct` | Base model |
| `BATCH_SIZE` | 8 | Per-device batch size |
| `GRAD_ACCUM_STEPS` | 4 | Gradient accumulation steps |
| `LEARNING_RATE` | 2e-5 | Learning rate |
| `LORA_RANK` | 16 | LoRA rank (memory vs quality) |
| `MAX_SEQ_LENGTH` | 2048 | Maximum sequence length |
| `MAX_STEPS` | 50 | Max steps per episode |

### Multi-GPU Training

For faster training in Colab Pro+:

```python
from accelerate import FullyShardedDataParallelPlugin, DistributedType
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
import torch

# Configure FSDP
fsdp_plugin = FullyShardedDataParallelPlugin(
    state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
    optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
)
accelerator = Accelerator(plugins=[fsdp_plugin])

# Wrap model and trainer accordingly
```

---

## Part 3: Action Schema Reference

### Actions

| Action | Target Required | Extra Fields | Description |
|--------|----------------|-------------|-------------|
| `read_logs` | Yes | `time_range_minutes` (optional, default 5) | Read logs for a service |
| `read_metrics` | Yes | — | Read metrics for a service |
| `read_deployment_history` | No | — | View recent deployments |
| `read_dependency_graph` | No | — | View service dependency graph |
| `identify_cause` | Yes | `hypothesis` (required) | Lock root cause hypothesis |
| `restart_pod` | Yes | — | Restart a pod (fix action) |
| `rollback` | Yes | — | Rollback deployment (fix action) |
| `scale_up` | Yes | — | Scale up pods (fix action) |
| `hotfix` | Yes | — | Apply config/service hotfix (fix action) |
| `escalate` | No | `justification` | Escalate to on-call engineer |
| `monitor_recovery` | No | — | Observe post-fix outcome |
| `resolve` | No | `justification` | Close the incident |

### Root Cause Hypotheses

```
memory_limit_too_low
bad_deployment
connection_pool_exhausted
traffic_spike
dependency_failure
config_error
redis_down
certificate_expired
```

### Root Cause → Correct Fix Mapping

```
memory_limit_too_low      → scale_up
bad_deployment           → rollback
connection_pool_exhausted→ hotfix
traffic_spike            → scale_up
dependency_failure       → restart_pod
config_error             → hotfix
redis_down               → restart_pod
certificate_expired      → hotfix
```

---

## Part 4: Using the Web UI

After deploying to Hugging Face Spaces:

1. Open your Space URL in a browser
2. Use the interactive UI to:
   - Reset the environment (start new incident)
   - Select an action and target service
   - View the observation response
   - Track reward accumulation
3. Useful for debugging and manual testing

---

## Part 5: Troubleshooting

### Docker Build Fails

```bash
# Verify Docker has sufficient memory
docker info | grep "Memory"

# Clean and rebuild
docker system prune -a
docker build -t incident_commander-env:latest -f server/Dockerfile . --no-cache
```

### Space Not Starting

Check the Space logs in the Hugging Face dashboard under "Settings" > "Logs".

Common issues:
- Missing `openenv.yaml` — re-run `openenv push`
- Port 8000 already in use — update `openenv.yaml` port

### Colab GPU Not Available

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

If `False`, go to Runtime > Change runtime type > T4 GPU.

### Training OOM (Out of Memory)

Reduce batch size and enable 4-bit quantization:

```python
# In training.py
BATCH_SIZE = 4          # reduced from 8
load_in_4bit = True      # already default
max_lora_rank = 8       # reduce from 16
gpu_memory_utilization = 0.4  # reduce from 0.6
```
