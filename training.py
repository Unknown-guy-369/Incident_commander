"""
AI Incident Commander — GRPO Training Pipeline (multi-turn).

Key differences from the earlier draft:
  * Uses a multi-turn rollout (not a single env step) per completion, so the
    four reward signals from rewards.py actually fire.
  * Uses `SyncEnvClient` (sync HTTP) — TRL's reward callbacks are synchronous,
    so the async openenv `EnvClient` would deadlock here.
  * Exposes the four reward signals + a small shaping bonus + format reward
    as separate `reward_funcs` so GRPOTrainer logs each one independently.
  * Curriculum-aware: starts at difficulty 1, can be bumped via env var.

Usage:
    INCIDENT_COMMANDER_ENV_URL=https://abishek-priyan-369-incident-commander.hf.space \
    python training.py --run --max-steps 200 --difficulty 1
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Optional

import torch
from datasets import Dataset

# Local modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rollout import (  # noqa: E402
    DEFAULT_REMOTE_URL,
    SYSTEM_PROMPT,
    SyncEnvClient,
    LocalEnvAdapter,
    rollout_episode,
    parse_action,
    parse_hypothesis,
    VALID_ACTIONS,
)
from rewards import make_grpo_reward_fns, format_reward_score  # noqa: E402


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = os.environ.get("INCIDENT_COMMANDER_MODEL", "unsloth/Llama-3.2-1B-Instruct")
MAX_SEQ_LENGTH = 2048
LORA_RANK = int(os.environ.get("INCIDENT_COMMANDER_LORA_RANK", "16"))
BATCH_SIZE = int(os.environ.get("INCIDENT_COMMANDER_BATCH", "2"))  # smaller default for T4
GRAD_ACCUM_STEPS = int(os.environ.get("INCIDENT_COMMANDER_GRAD_ACCUM", "4"))
LEARNING_RATE = float(os.environ.get("INCIDENT_COMMANDER_LR", "2e-5"))
NUM_GENERATIONS = int(os.environ.get("INCIDENT_COMMANDER_NUM_GEN", "4"))
ROLLOUT_MAX_STEPS = int(os.environ.get("INCIDENT_COMMANDER_ROLLOUT_STEPS", "12"))
REMOTE_ENV_URL = os.environ.get("INCIDENT_COMMANDER_ENV_URL", DEFAULT_REMOTE_URL).rstrip("/")
USE_LOCAL_ENV = os.environ.get("INCIDENT_COMMANDER_LOCAL", "0") == "1"


# ---------------------------------------------------------------------------
# Dataset — minimal "starting state" prompts
# ---------------------------------------------------------------------------

def generate_initial_prompts(num_samples: int = 100, difficulty: int = 1) -> Dataset:
    """Each prompt is just the initial alert + services overview. The model
    will be fed full transcripts at rollout time, so we don't need rich prompts
    at dataset-build time."""
    rows = []
    factory = (
        (lambda: LocalEnvAdapter(_make_local_env(), difficulty=difficulty))
        if USE_LOCAL_ENV
        else (lambda: SyncEnvClient(REMOTE_ENV_URL))
    )
    for i in range(num_samples):
        try:
            with factory() as env:
                init = env.reset(difficulty=difficulty)
                obs = init.get("observation", {})
            prompt = (
                f"System: {SYSTEM_PROMPT}\n\n"
                f"Initial alert: {obs.get('alert_summary', '')}\n"
                f"Services overview: {obs.get('services_overview', '')}\n\n"
                "Begin the investigation. What is your first action?"
            )
            rows.append({"prompt": prompt, "seed": i})
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] prompt {i} failed: {e}")
            continue
        if (i + 1) % 25 == 0:
            print(f"  generated {i + 1}/{num_samples} prompts")
    return Dataset.from_list(rows)


def _make_local_env():
    try:
        from server.incident_commander_environment import IncidentCommanderEnvironment
    except (ImportError, ModuleNotFoundError):
        from incident_commander.server.incident_commander_environment import (  # type: ignore
            IncidentCommanderEnvironment,
        )
    return IncidentCommanderEnvironment()


# ---------------------------------------------------------------------------
# Generation function used inside the rollout
# ---------------------------------------------------------------------------

def make_generate_fn(model, tokenizer, max_new_tokens: int = 200, temperature: float = 0.7):
    """Returns a sync `generate(prompt) -> completion` callable."""

    device = next(model.parameters()).device

    def _generate(prompt: str) -> str:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536).to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        completion = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        return completion

    return _generate


# ---------------------------------------------------------------------------
# Rollout-based reward function for GRPO
# ---------------------------------------------------------------------------

def make_episode_rollout(generate_fn, difficulty: int, max_steps: int):
    """Returns a `rollout_fn(prompt, completion) -> dict` for use with
    `make_grpo_reward_fns`."""

    factory = (
        (lambda: LocalEnvAdapter(_make_local_env(), difficulty=difficulty))
        if USE_LOCAL_ENV
        else (lambda: SyncEnvClient(REMOTE_ENV_URL))
    )

    def _rollout(prompt: str, completion: str) -> Dict[str, Any]:
        with factory() as env:
            state = rollout_episode(
                env, generate_fn,
                max_steps=max_steps,
                difficulty=difficulty,
            )
        return state.to_dict()

    return _rollout


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="Run the training pipeline")
    parser.add_argument("--max-steps", type=int, default=int(os.environ.get("INCIDENT_COMMANDER_MAX_STEPS", "200")),
                        help="GRPO training steps (default 200)")
    parser.add_argument("--difficulty", type=int, default=1, choices=[1, 2, 3, 4])
    parser.add_argument("--num-prompts", type=int, default=200,
                        help="Initial prompt dataset size")
    parser.add_argument("--rollout-steps", type=int, default=ROLLOUT_MAX_STEPS,
                        help="Max env steps inside each rollout")
    parser.add_argument("--output-dir", type=str, default="outputs/commander_grpo")
    args = parser.parse_args()

    if not args.run:
        print("Training script ready. Run with `python training.py --run`.")
        return

    # Lazy imports — these are heavy and only needed when actually running.
    from unsloth import FastLanguageModel, PatchFastRL  # type: ignore
    PatchFastRL("GRPO", FastLanguageModel)
    from trl import GRPOConfig, GRPOTrainer  # type: ignore

    print(f"Loading {MODEL_NAME} via Unsloth (4-bit)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        fast_inference=False,  # vLLM crashes on T4 (compute capability 7.5)
        max_lora_rank=LORA_RANK,
        gpu_memory_utilization=0.6,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=LORA_RANK,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    print("Generating initial prompts...")
    dataset = generate_initial_prompts(num_samples=args.num_prompts, difficulty=args.difficulty)

    print("Building rollout-based reward functions...")
    generate_fn = make_generate_fn(model, tokenizer, max_new_tokens=180, temperature=0.7)
    rollout_fn = make_episode_rollout(generate_fn, args.difficulty, args.rollout_steps)
    reward_fns = make_grpo_reward_fns(rollout_fn)

    print("Configuring GRPOTrainer...")
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        max_prompt_length=1024,
        max_completion_length=256,
        num_generations=NUM_GENERATIONS,
        max_steps=args.max_steps,
        logging_steps=5,
        save_steps=max(20, args.max_steps // 10),
        beta=0.04,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        report_to=os.environ.get("INCIDENT_COMMANDER_REPORT_TO", "none"),
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fns,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print(f"Starting GRPO training ({args.max_steps} steps, difficulty {args.difficulty})...")
    trainer.train()

    final_dir = os.path.join(args.output_dir, "..", "commander_final")
    final_dir = os.path.normpath(final_dir)
    os.makedirs(final_dir, exist_ok=True)
    print(f"Saving merged model to {final_dir}...")
    model.save_pretrained_merged(final_dir, tokenizer, save_method="merged_16bit")
    print("Training complete.")


if __name__ == "__main__":
    main()
