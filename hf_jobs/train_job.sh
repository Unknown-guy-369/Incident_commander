#!/usr/bin/env bash
set -euo pipefail

# Runs inside a Hugging Face Job container.
# It clones the repo, installs training dependencies, and starts GRPO training.

REPO_URL="${REPO_URL:-https://github.com/Unknown-guy-369/Incident_commander.git}"
REPO_REF="${REPO_REF:-main}"
WORKDIR="${WORKDIR:-/workspace/incident_commander}"

INCIDENT_COMMANDER_ENV_URL="${INCIDENT_COMMANDER_ENV_URL:-https://abishek-priyan-369-incident-commander.hf.space}"
INCIDENT_COMMANDER_MAX_STEPS="${INCIDENT_COMMANDER_MAX_STEPS:-200}"
INCIDENT_COMMANDER_DIFFICULTY="${INCIDENT_COMMANDER_DIFFICULTY:-1}"
INCIDENT_COMMANDER_NUM_PROMPTS="${INCIDENT_COMMANDER_NUM_PROMPTS:-80}"
INCIDENT_COMMANDER_ROLLOUT_STEPS="${INCIDENT_COMMANDER_ROLLOUT_STEPS:-8}"
INCIDENT_COMMANDER_OUTPUT_DIR="${INCIDENT_COMMANDER_OUTPUT_DIR:-outputs/commander_grpo}"
INCIDENT_COMMANDER_STEPS_PER_BATCH="${INCIDENT_COMMANDER_STEPS_PER_BATCH:-10}"
INCIDENT_COMMANDER_BATCHES="${INCIDENT_COMMANDER_BATCHES:-1}"

if [ "${INCIDENT_COMMANDER_STEPS_PER_BATCH}" -gt 0 ]; then
  INCIDENT_COMMANDER_BATCHES=$(( INCIDENT_COMMANDER_MAX_STEPS / INCIDENT_COMMANDER_STEPS_PER_BATCH ))
  if [ "${INCIDENT_COMMANDER_BATCHES}" -lt 1 ]; then
    INCIDENT_COMMANDER_BATCHES=1
  fi
fi

echo "==> Cloning repo: ${REPO_URL} (ref: ${REPO_REF})"
if ! command -v git >/dev/null 2>&1; then
  echo "==> git not found; installing git"
  if command -v apt-get >/dev/null 2>&1; then
    apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
  else
    echo "No supported package manager found to install git."
    exit 1
  fi
fi
rm -rf "${WORKDIR}"
git clone --depth 1 --branch "${REPO_REF}" "${REPO_URL}" "${WORKDIR}"
cd "${WORKDIR}"

echo "==> Installing base dependencies"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "==> Installing training dependencies"
# Unsloth is intentionally installed separately to keep app/runtime deps isolated.
python -m pip install "unsloth>=2025.10.0"

echo "==> Launching notebook-equivalent training script"
export INCIDENT_COMMANDER_ENV_URL
python colab/incident_commander_training_job.py --run \
  --difficulty "${INCIDENT_COMMANDER_DIFFICULTY}" \
  --num-prompts "${INCIDENT_COMMANDER_NUM_PROMPTS}" \
  --rollout-steps "${INCIDENT_COMMANDER_ROLLOUT_STEPS}" \
  --steps-per-batch "${INCIDENT_COMMANDER_STEPS_PER_BATCH}" \
  --batches "${INCIDENT_COMMANDER_BATCHES}" \
  --output-dir "${INCIDENT_COMMANDER_OUTPUT_DIR}"

echo "==> Training finished"
echo "Artifacts are under: ${WORKDIR}/${INCIDENT_COMMANDER_OUTPUT_DIR} and ${WORKDIR}/outputs/commander_final"
