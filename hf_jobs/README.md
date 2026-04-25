# Hugging Face Jobs Training

This folder runs the notebook training flow on HF Jobs (without using `training.py`).

## Files

- `submit_job.ps1` - submit a training job from Windows PowerShell.
- `submit_job.sh` - submit a training job from bash/zsh.
- `train_job.sh` - training entry script used inside a job container.
- `colab/incident_commander_training_job.py` - notebook-equivalent training logic.

Note: some HF job images do not include `git`. The scripts now auto-install it
via `apt-get` before cloning.

## Prerequisites

1. Install the `hf` CLI.
2. Authenticate:

```bash
hf auth login
```

## Quick Start (PowerShell)

```powershell
cd C:\meta-hackathon\final-round\incident_commander
.\hf_jobs\submit_job.ps1
```

## Quick Start (bash)

```bash
cd incident_commander
bash hf_jobs/submit_job.sh
```

## Common overrides

Set these env vars before running the submit script:

- `FLAVOR` (default: `a10g-small`)
- `IMAGE` (default: `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel`)
- `INCIDENT_COMMANDER_ENV_URL` (default points to your Space)
- `INCIDENT_COMMANDER_MAX_STEPS` (default: `200`)
- `INCIDENT_COMMANDER_DIFFICULTY` (default: `1`)
- `INCIDENT_COMMANDER_NUM_PROMPTS` (default: `80`)
- `INCIDENT_COMMANDER_ROLLOUT_STEPS` (default: `8`)
- `INCIDENT_COMMANDER_STEPS_PER_BATCH` (default: `10`)
- `INCIDENT_COMMANDER_OUTPUT_DIR` (default: `outputs/commander_grpo`)
- `REPO_URL` / `REPO_REF` (if training from a different branch/repo)

`INCIDENT_COMMANDER_MAX_STEPS` is translated into notebook-style batches using:

`batches = max(1, MAX_STEPS / STEPS_PER_BATCH)`

After submission:

```bash
hf jobs
hf jobs logs <job-id>
```
