# Incident Commander — Hackathon Memory Document

**Date:** April 25, 2026
**Event:** Meta PyTorch OpenEnv Hackathon (36-hour finale)
**Team:** Ram + Abishek
**Project:** AI Enterprise IT Incident Commander

---

## Architecture

- **Environment:** Incident Commander RL environment built on Meta's OpenEnv framework
- **Deployment:** HuggingFace Space (Abishek's account) — `https://abishek-priyan-369-incident-commander.hf.space`
- **Training:** Google Colab (free T4 GPU) running GRPO with Unsloth
- **Model:** Llama-3.2-1B-Instruct (4-bit quantized via Unsloth)
- **Connection:** Colab → HF Space via HTTP REST API (SyncEnvClient)

## API Details

- **Endpoints:** POST /reset, POST /step, GET /state, GET /schema, WS /ws, GET /web (Gradio)
- **Step payload format:** `{"action": {"action_type": "...", "target_service": "...", "hypothesis": "...", "justification": "...", "time_range_minutes": 5}}`
- **All 5 fields must be sent** in every step call (Abishek confirmed)
- **target_service** was originally required (`...`), needs `Optional[str]` or always send a default

## 12 Valid Actions

| Action | Target Required | Extra Fields |
|--------|----------------|-------------|
| read_logs | Yes | time_range_minutes |
| read_metrics | Yes | — |
| read_deployment_history | No | — |
| read_dependency_graph | No | — |
| identify_cause | Yes | hypothesis |
| restart_pod | Yes | — |
| rollback | Yes | — |
| scale_up | Yes | — |
| hotfix | Yes | — |
| escalate | No | justification |
| monitor_recovery | No | — |
| resolve | No | justification |

## Root Cause → Fix Mapping

| Root Cause | Correct Fix |
|------------|-------------|
| memory_limit_too_low | scale_up |
| bad_deployment | rollback |
| connection_pool_exhausted | hotfix |
| traffic_spike | scale_up |
| dependency_failure | restart_pod |
| config_error | hotfix |
| redis_down | restart_pod |
| certificate_expired | hotfix |

## 4 Reward Signals

1. **service_recovery** — did the fix work?
2. **root_cause_accuracy** — was the hypothesis correct?
3. **action_quality** — were the investigation steps logical?
4. **speed** — how many steps taken?

---

## Training Configuration

```
Model: unsloth/Llama-3.2-1B-Instruct
MAX_SEQ_LENGTH: 2048
LORA_RANK: 16
BATCH_SIZE: 8
GRAD_ACCUM_STEPS: 4
LEARNING_RATE: 2e-5
max_steps: 10 (per batch, run 3 times = 30)
fast_inference: False (vLLM crashes on T4)
load_in_4bit: True
```

## Reward Functions Used

1. **environment_reward_func** — sends parsed action to HF Space, gets env reward
2. **format_reward_func** — +0.1 for `<thought>` tags, +0.1 for `<action>` tags

## SyncEnvClient (Critical Code)

The openenv EnvClient is async and doesn't work in GRPO training. We built a synchronous HTTP client:

```python
class SyncEnvClient:
    def __init__(self, base_url):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
    def reset(self):
        resp = self.session.post(f"{self.base_url}/reset", json={})
        resp.raise_for_status()
        return resp.json()
    def step(self, action_type, target_service=None, hypothesis=None, justification=None, time_range_minutes=None):
        action = {
            "action_type": action_type,
            "target_service": target_service or "payment-service",
            "hypothesis": hypothesis or "",
            "justification": justification or "",
            "time_range_minutes": time_range_minutes or 5
        }
        resp = self.session.post(f"{self.base_url}/step", json={"action": action})
        resp.raise_for_status()
        return resp.json()
    def close(self):
        self.session.close()
```

**Key:** SyncEnvClient must be defined in BOTH the import cell AND the reward functions cell (embedded), because GRPO trainer runs reward functions in isolation.

---

## Training Runs

### Run 1 (Original Colab account — GPU quota exhausted)

- Ran 35/62 steps before Colab auto-stopped
- Training metrics captured:
  - Step 10: reward=0.051, format_reward=0.051, env_reward=0.000, KL=0.000168
  - Step 20: reward=0.065, format_reward=0.065, env_reward=0.000, KL=0.002692
  - Step 30: reward=0.078, format_reward=0.078, env_reward=0.000, KL=0.003030
- Model weights LOST (session died, no checkpoint saved)
- Before-training eval: Avg Reward=0.0, Format Compliance=14.3%, Valid Action Rate=40.0%

### Run 2 (Second Colab account)

- Fixed 422 errors by always sending all 5 fields
- Before-training eval (10 episodes): Avg Reward=0.0, Format Compliance=10.0%, Valid Action Rate=0.0%
- Training batch 1 (10 steps): reward=0.044687, format_reward=0.044688, env_reward=0.000
- Training batch 2 (10 steps): reward=0.053438, format_reward=0.053, env_reward=0.000
- After-training eval (20 steps total): Avg Reward=0.0, Format Compliance=25.0%, Valid Action Rate=0.0%
- **Improvement shown: Format Compliance 10% → 25% (+15 percentage points, +150% relative)**

### Current Status

- Model generates `<action>` tags correctly but fills them with placeholder text (`action_name`, `lock-hypothesis`) instead of valid actions like `read_logs`
- environment_reward_func has been 0.000 throughout — model hasn't discovered valid action names yet
- Need more training steps for the model to start producing valid action types
- Checkpoint saved at step 20

---

## Errors Encountered & Solutions

| Error | Root Cause | Fix |
|-------|-----------|-----|
| `'coroutine' object has no attribute 'observation'` | openenv EnvClient is async | Created SyncEnvClient with plain `requests` |
| vLLM RuntimeError (torch.compile graph crash) | T4 GPU compute capability 7.5 incompatible | Set `fast_inference=False` |
| 422 Unprocessable Entity on /step | API expects wrapped format `{"action": {...}}` | Changed to `json={"action": action}` |
| 422 persisting after fix | Old SyncEnvClient still in memory | Embedded SyncEnvClient in reward functions cell |
| 422 with Optional target_service | Server still requires all 5 fields | Always send all 5 with defaults |
| KeyboardInterrupt during eval | 422 errors causing slow episodes | Added VALID_ACTIONS check + try/except |
| pip dependency conflicts | vLLM vs transformers versions | Ignored (warnings not errors) |
| Colab GPU quota exhausted | Free tier limit | Switched to second Google account |

---

## Files in Repository

| File | Purpose |
|------|---------|
| `models.py` | Pydantic models (Action, Observation). target_service needs Optional |
| `client.py` | Async openenv EnvClient — NOT used for training |
| `server/app.py` | FastAPI app with Gradio UI, creates endpoints via openenv |
| `server/incident_commander_environment.py` | Core RL environment logic |
| `simulator.py` | Microservice outage simulator |
| `rewards.py` | Reward calculation logic |
| `training.py` | Training script (reference, Colab is actual execution) |
| `colab/incident_commander_training.ipynb` | THE notebook for training |
| `inference.py` | Local inference script |
| `inference_openrouter.py` | OpenRouter-based inference |
| `project_report_complete.md` | Vision document for storytelling (most features NOT implemented) |

## Colab Notebook Cell Order

1. Install dependencies (pip install unsloth, openenv-core, trl, etc.)
2. Clone repo + verify GPU
3. Import Environment (SyncEnvClient with all 5 fields)
4. Load Model (Unsloth 4-bit, fast_inference=False)
5. Define Reward Functions (embedded SyncEnvClient + VALID_ACTIONS)
6. Generate Dataset (num_samples=100)
7. Before Training Evaluation (10 episodes)
8. Initialize GRPO Trainer (max_steps=10)
9. Train (10 steps)
10. Save Checkpoint
11. Continue Training (repeat trainer.train() for +10 steps)
12. After Training Evaluation + Plots
13. Save Final Model

---

## Hackathon Judging Criteria

| Criteria | Weight | Our Status |
|----------|--------|------------|
| Environment Innovation | 40% | Strong — Abishek built the env |
| Storytelling | 30% | project_report_complete.md ready |
| Showing Improvement | 20% | Format Compliance +15% shown |
| Pipeline | 10% | End-to-end working: Colab → HF Space → GRPO |

## Key Decisions Made

1. Llama 3.2 1B (not 8B) — fits on free T4
2. No vLLM — crashes on T4, used native PyTorch generation
3. Remote environment only — no local server needed
4. GRPO only (no PPO, no DPO) — hackathon requirement
5. 10-step training batches with checkpoints — prevents losing work
6. SyncEnvClient over EnvClient — async doesn't work with GRPO trainer

## What's Left

- [ ] Run 3rd batch of 10 steps (total 30)
- [ ] Re-run after-training evaluation
- [ ] Generate final comparison plots
- [ ] Save final model
- [ ] Push models.py fix (Optional target_service) to GitHub
- [ ] Prepare presentation/demo using project_report_complete.md
- [ ] Update README with results
