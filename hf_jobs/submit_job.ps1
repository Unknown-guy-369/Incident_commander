$ErrorActionPreference = "Stop"

# Local helper for Windows/PowerShell to submit a Hugging Face Job.
# Prerequisites:
#   1) hf CLI installed
#   2) hf auth login completed

$Flavor = if ($env:FLAVOR) { $env:FLAVOR } else { "a10g-small" }
$Image = if ($env:IMAGE) { $env:IMAGE } else { "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel" }

$RepoUrl = if ($env:REPO_URL) { $env:REPO_URL } else { "https://github.com/Unknown-guy-369/Incident_commander.git" }
$RepoRef = if ($env:REPO_REF) { $env:REPO_REF } else { "main" }

$RemoteEnvUrl = if ($env:INCIDENT_COMMANDER_ENV_URL) { $env:INCIDENT_COMMANDER_ENV_URL } else { "https://abishek-priyan-369-incident-commander.hf.space" }
$MaxSteps = if ($env:INCIDENT_COMMANDER_MAX_STEPS) { $env:INCIDENT_COMMANDER_MAX_STEPS } else { "200" }
$Difficulty = if ($env:INCIDENT_COMMANDER_DIFFICULTY) { $env:INCIDENT_COMMANDER_DIFFICULTY } else { "1" }
$NumPrompts = if ($env:INCIDENT_COMMANDER_NUM_PROMPTS) { $env:INCIDENT_COMMANDER_NUM_PROMPTS } else { "80" }
$RolloutSteps = if ($env:INCIDENT_COMMANDER_ROLLOUT_STEPS) { $env:INCIDENT_COMMANDER_ROLLOUT_STEPS } else { "8" }
$StepsPerBatch = if ($env:INCIDENT_COMMANDER_STEPS_PER_BATCH) { $env:INCIDENT_COMMANDER_STEPS_PER_BATCH } else { "10" }
$OutputDir = if ($env:INCIDENT_COMMANDER_OUTPUT_DIR) { $env:INCIDENT_COMMANDER_OUTPUT_DIR } else { "outputs/commander_grpo" }
$Batches = [Math]::Max([int]($MaxSteps) / [int]($StepsPerBatch), 1)

$JobCommand = @(
  "set -e",
  "if ! command -v git >/dev/null 2>&1; then apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*; fi",
  "git clone --depth 1 --branch '${RepoRef}' '${RepoUrl}' /workspace/incident_commander",
  "cd /workspace/incident_commander",
  "python -m pip install --upgrade pip",
  "python -m pip install -r requirements.txt",
  "python -m pip install 'unsloth>=2025.10.0'",
  "INCIDENT_COMMANDER_ENV_URL='${RemoteEnvUrl}' python colab/incident_commander_training_job.py --run --difficulty ${Difficulty} --num-prompts ${NumPrompts} --rollout-steps ${RolloutSteps} --steps-per-batch ${StepsPerBatch} --batches ${Batches} --output-dir '${OutputDir}'"
) -join "; "

Write-Host "Submitting HF Job with flavor=$Flavor image=$Image"

hf jobs run `
  --flavor $Flavor `
  $Image `
  -- `
  bash -lc "$JobCommand"
