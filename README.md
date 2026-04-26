---
title: Incident Commander Environment
emoji: 🚨
colorFrom: purple
colorTo: red
sdk: docker
pinned: false
app_port: 8000
base_path: /
tags:
  - openenv
  - rl-environment
  - sre
  - causal-reasoning
  - theme-3-1
---

# AI Incident Commander

An OpenEnv RL environment that trains LLMs to investigate and resolve
microservice outages. Theme **3.1 — Professional Tasks**: real stateful
tool interactions, partial observability, causal-chain scenarios,
multi-round fix loops, and four anti-shortcut reward signals.

> **Live environment:** [`abishek-priyan-369-incident-commander.hf.space`](https://abishek-priyan-369-incident-commander.hf.space)
> **Training notebook:** [`colab/incident_commander_training.ipynb`](colab/incident_commander_training.ipynb)
> **Results plots:** [`assets/before_after_comparison.png`](assets/before_after_comparison.png) · [`assets/per_signal_comparison.png`](assets/per_signal_comparison.png)

---

## 1. Why this environment exists

Most LLM-as-agent benchmarks reward the model for *answering* well. SRE
incident response rewards an agent for *acting* well: first investigate,
then commit to a hypothesis, then apply a matching fix, then verify the
fix worked, and only then close the incident. Skip any of those steps
and the system stays broken.

Static prompt → answer datasets cannot teach this. Incident Commander
gives the model:

- **Partial observability** — only the alert + service overview is
  visible at reset. Logs, metrics, deployment history and the dependency
  graph are hidden until the agent explicitly fetches them.
- **Causal chains** — Redis down → Auth slow → API timeouts → Payment
  failing. Symptoms are downstream; the agent must trace the chain back.
- **Dynamic randomisation** — every episode reshuffles CPU %, error
  rates, log timestamps and red-herring services. You cannot memorise
  "CPU > 90 % means traffic spike."
- **Anti-shortcut enforcement** — fix actions are *blocked* until the
  agent has read at least 3 distinct log/metric sources AND committed a
  hypothesis. `resolve` is blocked until the post-fix status is
  `recovered`.
- **Four reward signals** logged independently:
  service recovery, root-cause accuracy, action quality, speed.

## 2. Quick start (client side)

```python
from rollout import SyncEnvClient

with SyncEnvClient("https://abishek-priyan-369-incident-commander.hf.space") as env:
    obs = env.reset(difficulty=1)["observation"]
    print(obs["alert_summary"])  # e.g. "[CRITICAL] PagerDuty: payment-service error rate critical"

    # Investigate
    env.step(action_type="read_logs", target_service="payment-service")
    env.step(action_type="read_metrics", target_service="payment-service")
    env.step(action_type="read_deployment_history")

    # Lock hypothesis
    env.step(action_type="identify_cause",
             target_service="payment-service",
             hypothesis="memory_limit_too_low")

    # Apply matching fix
    env.step(action_type="scale_up", target_service="payment-service")

    # Verify and close
    env.step(action_type="monitor_recovery")
    final = env.step(action_type="resolve", justification="service recovered")
    print("Final reward:", final["reward"])
    print("Breakdown:",   final["observation"]["reward_breakdown"])
```

## 3. Action schema

| Action | Target | Extra | Purpose |
|---|---|---|---|
| `read_logs` | yes | `time_range_minutes` | Fetch recent logs |
| `read_metrics` | yes | — | CPU / memory / error rate / p99 |
| `read_deployment_history` | no | — | Recent deploys |
| `read_dependency_graph` | no | — | Service-call graph |
| `identify_cause` | yes | `hypothesis` | Lock root-cause hypothesis |
| `restart_pod` | yes | — | Restart the pod |
| `rollback` | yes | — | Revert latest deploy |
| `scale_up` | yes | — | Add compute |
| `hotfix` | yes | — | Patch config / cert / pool |
| `escalate` | no | `justification` | Page a human |
| `monitor_recovery` | no | — | Observe post-fix status |
| `resolve` | no | `justification` | Close the incident |

### Root cause → correct fix

| Root cause | Fix |
|---|---|
| `memory_limit_too_low` | `scale_up` |
| `bad_deployment` | `rollback` |
| `connection_pool_exhausted` | `hotfix` |
| `traffic_spike` | `scale_up` |
| `dependency_failure` | `restart_pod` |
| `config_error` | `hotfix` |
| `redis_down` | `restart_pod` |
| `certificate_expired` | `hotfix` |

## 4. Reward design

All four signals fire only when the episode terminates (resolve or timeout).
They are exposed individually to GRPO via `rewards.make_grpo_reward_fns(...)`
so the trainer logs each one independently.

| Signal | Range | What it captures |
|---|---|---|
| `service_recovery` | -20 .. +30 | Did the fix work? `+30 recovered`, `+5 degraded`, `-15 worse`, `-20 no fix applied` |
| `root_cause_accuracy` | -15 .. +25 | `+25` correct hypothesis locked *before* fix; `-10` wrong; `-15` no hypothesis |
| `action_quality` | -85 .. +5 | Anti-shortcuts: fix-without-investigation `-15`, fix-mismatches-hypothesis `-20`, rollback-spam `-50`, escalation-spam `-30`, resolve-without-reading `-25`. Bonus `+5` for log+metric coverage |
| `speed` | 0 .. +15 | `(steps_remaining/50) × 15` — paid only on `recovered` |

The clamped aggregate `[0, 1]` is what the env's `step` returns. For RL
training, prefer `compute_signed_reward(...)` (also in `rewards.py`),
which keeps negative advantage so the gradient does not collapse.

## 5. Project layout

```
incident_commander/
├── README.md                 # this file
├── DEPLOY.md                 # HF Space + Colab deployment guide
├── REVIEW.md                 # internal review notes
├── HACKATHON_MEMORY.md       # team memory document
├── pyproject.toml
├── requirements.txt          # local dev / training deps
├── openenv.yaml              # OpenEnv manifest
├── Dockerfile                # HF Space build (root)
├── app.py                    # uvicorn entry point used by Dockerfile
│
├── __init__.py
├── client.py                 # IncidentCommanderEnv (async openenv client)
├── rollout.py                # SyncEnvClient + multi-turn rollout
├── models.py                 # Pydantic Action / Observation
├── simulator.py              # 8 scenarios + dynamic log/metric generators
├── rewards.py                # 4 reward signals + GRPO factory
├── training.py               # CLI training driver
├── inference.py              # local trained-model evaluation
├── inference_openrouter.py   # OpenRouter (Claude / GPT / Gemini) baseline
│
├── server/
│   ├── app.py                # FastAPI + Gradio /web UI
│   ├── incident_commander_environment.py
│   ├── Dockerfile            # alternative server-only image
│   ├── requirements.txt      # runtime-only deps for HF Space
│   └── __init__.py
│
├── colab/
│   └── incident_commander_training.ipynb
│
├── tests/
│   ├── smoke_test.py
│   ├── test_env.py
│   └── test_training.py
│
└── assets/                   # plots produced by the notebook
    ├── before_after_comparison.png
    └── per_signal_comparison.png
```

## 6. Running the training

### A. Colab (recommended)

Open `colab/incident_commander_training.ipynb`, run cells top-to-bottom.
The notebook:

1. Installs pinned deps (Unsloth + TRL + OpenEnv-core).
2. Loads `unsloth/Llama-3.2-1B-Instruct` in 4-bit + LoRA r=16.
3. Builds **multi-turn rollout reward functions** — each completion runs
   a full episode against the live HF Space.
4. Runs a 15-episode evaluation **before** training.
5. Trains with `GRPOTrainer` for 200 steps on difficulty 1.
6. Saves the merged model and runs a 15-episode evaluation **after**.
7. Saves both before/after plots to `assets/`.

### B. Locally

```bash
# Train
INCIDENT_COMMANDER_ENV_URL=https://abishek-priyan-369-incident-commander.hf.space \
python training.py --run --max-steps 200 --difficulty 1

# Evaluate
python inference.py --model outputs/commander_final --episodes 20

# Sanity-check the env logic
python tests/smoke_test.py
pytest tests/test_env.py -v
```

## 7. Results

> Plots are produced by Cell 11 of the Colab notebook; embed once you
> have a real run.

![Before vs After GRPO (avg reward, resolve rate, root-cause accuracy)](assets/before_after_comparison.png)

![Per-signal reward — before vs after](assets/per_signal_comparison.png)

| Metric | Before | After | Δ |
|---|---|---|---|
| Avg episode reward | _fill in_ | _fill in_ | _fill in_ |
| Resolve rate | _fill in_ | _fill in_ | _fill in_ |
| Root-cause accuracy | _fill in_ | _fill in_ | _fill in_ |

(See `colab/incident_commander_training.ipynb` Cell 11 for the script
that generated these.)

## 8. What makes this submission distinctive

- **Anti-shortcut enforcement is in the environment, not just the reward.**
  Fix actions are *blocked* (not merely penalised) when investigation is
  insufficient, and `resolve` is blocked unless `monitor_recovery`
  reported `recovered`. An LLM cannot bypass these by clever output
  formatting — the env is the verifier.
- **Four independently-logged reward channels.** Per the
  *What-Judges-Look-For* guide: "OpenEnv's Rubric system thoughtfully
  (composable rubrics > monolithic scoring)." Our `make_grpo_reward_fns`
  exposes one TRL reward function per signal so wandb / TRL logs show
  four curves out of the box.
- **Dynamic scenarios.** 8 root cause types × randomised numbers ×
  red-herring services per episode. Memorising symptom→fix is impossible.
- **Multi-hop causal chains.** `redis_down` propagates Redis→Auth→API.
  The agent must trace the dependency graph upstream.

## 9. Links

- Live env (HF Space): `https://abishek-priyan-369-incident-commander.hf.space`
  - `/web` — interactive Gradio UI
  - `/docs` — OpenAPI schema
  - `/health` — health check
- Training notebook: [`colab/incident_commander_training.ipynb`](colab/incident_commander_training.ipynb)
- Deployment guide: [`DEPLOY.md`](DEPLOY.md)

## 10. Citing / acknowledgements

Built on:

- [OpenEnv](https://github.com/meta-pytorch/OpenEnv) (Meta PyTorch)
- [TRL GRPO Trainer](https://huggingface.co/docs/trl/main/en/grpo_trainer) (Hugging Face)
- [Unsloth](https://github.com/unslothai/unsloth) for memory-efficient fine-tuning
