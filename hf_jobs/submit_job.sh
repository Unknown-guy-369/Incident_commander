#!/usr/bin/env bash
set -euo pipefail

# Local helper script: submit a Hugging Face Job for GRPO training.
# Prerequisites:
#   1) `hf` CLI installed
#   2) `hf auth login` completed

FLAVOR="${FLAVOR:-a10g-small}"
IMAGE="${IMAGE:-pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel}"

REPO_URL="${REPO_URL:-https://github.com/Unknown-guy-369/Incident_commander.git}"
REPO_REF="${REPO_REF:-main}"

INCIDENT_COMMANDER_ENV_URL="${INCIDENT_COMMANDER_ENV_URL:-https://abishek-priyan-369-incident-commander.hf.space}"
INCIDENT_COMMANDER_MAX_STEPS="${INCIDENT_COMMANDER_MAX_STEPS:-200}"
INCIDENT_COMMANDER_DIFFICULTY="${INCIDENT_COMMANDER_DIFFICULTY:-1}"
INCIDENT_COMMANDER_NUM_PROMPTS="${INCIDENT_COMMANDER_NUM_PROMPTS:-80}"
INCIDENT_COMMANDER_ROLLOUT_STEPS="${INCIDENT_COMMANDER_ROLLOUT_STEPS:-8}"
INCIDENT_COMMANDER_STEPS_PER_BATCH="${INCIDENT_COMMANDER_STEPS_PER_BATCH:-10}"
INCIDENT_COMMANDER_OUTPUT_DIR="${INCIDENT_COMMANDER_OUTPUT_DIR:-outputs/commander_grpo}"

SCRIPT_PATH="/workspace/train_job.sh"

echo "Submitting HF Job with flavor=${FLAVOR}, image=${IMAGE}"

hf jobs run \
  --flavor "${FLAVOR}" \
  --env "REPO_URL=${REPO_URL}" \
  --env "REPO_REF=${REPO_REF}" \
  --env "INCIDENT_COMMANDER_ENV_URL=${INCIDENT_COMMANDER_ENV_URL}" \
  --env "INCIDENT_COMMANDER_MAX_STEPS=${INCIDENT_COMMANDER_MAX_STEPS}" \
  --env "INCIDENT_COMMANDER_DIFFICULTY=${INCIDENT_COMMANDER_DIFFICULTY}" \
  --env "INCIDENT_COMMANDER_NUM_PROMPTS=${INCIDENT_COMMANDER_NUM_PROMPTS}" \
  --env "INCIDENT_COMMANDER_ROLLOUT_STEPS=${INCIDENT_COMMANDER_ROLLOUT_STEPS}" \
  --env "INCIDENT_COMMANDER_OUTPUT_DIR=${INCIDENT_COMMANDER_OUTPUT_DIR}" \
  "${IMAGE}" \
  -- \
  bash -lc "cat > ${SCRIPT_PATH} << 'EOF'
$(cat "$(dirname "$0")/train_job.sh")
EOF
chmod +x ${SCRIPT_PATH}
${SCRIPT_PATH}"
