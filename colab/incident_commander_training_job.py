"""
Notebook-equivalent training entrypoint for Hugging Face Jobs.

This script follows the workflow in `colab/incident_commander_training.ipynb`
and is intended for non-interactive training runs.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, Any

import torch
from datasets import Dataset
from unsloth import FastLanguageModel, PatchFastRL

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

PatchFastRL("GRPO", FastLanguageModel)
from trl import GRPOConfig, GRPOTrainer

from rollout import SyncEnvClient, DEFAULT_REMOTE_URL, rollout_episode, SYSTEM_PROMPT
from rewards import make_grpo_reward_fns


def make_generate_fn(model, tokenizer, max_new_tokens: int = 160, temperature: float = 0.7):
    device = next(model.parameters()).device

    def _generate(prompt: str) -> str:
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1536
        ).to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        return tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

    return _generate


def generate_initial_prompts(env_url: str, num_samples: int, difficulty: int) -> Dataset:
    rows = []
    for i in range(num_samples):
        try:
            with SyncEnvClient(env_url) as env:
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
            print(f"[warn] prompt {i} failed: {e}")
        if (i + 1) % 25 == 0:
            print(f"generated {i + 1}/{num_samples}")
    return Dataset.from_list(rows)


def make_episode_rollout(env_url: str, generate_fn, rollout_steps: int, difficulty: int):
    def _rollout(prompt: str, completion: str) -> Dict[str, Any]:
        with SyncEnvClient(env_url) as env:
            state = rollout_episode(
                env, generate_fn, max_steps=rollout_steps, difficulty=difficulty
            )
        return state.to_dict()

    return _rollout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--model-name", type=str, default="unsloth/Llama-3.2-1B-Instruct")
    parser.add_argument("--output-dir", type=str, default="outputs/commander_grpo")
    parser.add_argument("--difficulty", type=int, default=1, choices=[1, 2, 3, 4])
    parser.add_argument("--num-prompts", type=int, default=80)
    parser.add_argument("--rollout-steps", type=int, default=8)
    parser.add_argument("--steps-per-batch", type=int, default=10)
    parser.add_argument("--batches", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--num-generations", type=int, default=2)
    args = parser.parse_args()

    if not args.run:
        print("Notebook training script ready. Run with --run.")
        return

    env_url = os.environ.get("INCIDENT_COMMANDER_ENV_URL", DEFAULT_REMOTE_URL).rstrip("/")
    print(f"Using env: {env_url}")

    with SyncEnvClient(env_url) as env:
        sanity = env.reset(difficulty=args.difficulty)
    print("Env sanity check OK:", bool(sanity.get("observation")))

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=2048,
        dtype=torch.float16,
        load_in_4bit=True,
        fast_inference=False,
        max_lora_rank=16,
        gpu_memory_utilization=0.6,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    dataset = generate_initial_prompts(env_url, args.num_prompts, args.difficulty)
    print("Dataset size:", len(dataset))

    generate_fn = make_generate_fn(model, tokenizer)
    rollout_fn = make_episode_rollout(env_url, generate_fn, args.rollout_steps, args.difficulty)
    reward_fns = make_grpo_reward_fns(rollout_fn)
    print("Reward fns:", [fn.__name__ for fn in reward_fns])

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        max_prompt_length=1024,
        max_completion_length=200,
        num_generations=args.num_generations,
        max_steps=args.steps_per_batch,
        logging_steps=2,
        save_steps=args.steps_per_batch,
        save_total_limit=4,
        beta=0.04,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        report_to="none",
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fns,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print(f"Training batch 1/{args.batches} for {args.steps_per_batch} steps")
    trainer.train()
    total_steps = args.steps_per_batch

    for i in range(2, args.batches + 1):
        training_args.max_steps += args.steps_per_batch
        print(f"Training batch {i}/{args.batches} to step {training_args.max_steps}")
        trainer.train(resume_from_checkpoint=True)
        total_steps = training_args.max_steps

    ckpt_dir = os.path.join(args.output_dir, f"manual_step_{total_steps}")
    os.makedirs(ckpt_dir, exist_ok=True)
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)

    final_dir = "outputs/commander_final"
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained_merged(final_dir, tokenizer, save_method="merged_16bit")
    print("Saved merged model to", final_dir)


if __name__ == "__main__":
    main()
