# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "unsloth",
#     "trl>=0.12.0,<0.14.0",
#     "datasets>=2.20.0",
#     "peft>=0.11.0",
#     "accelerate>=0.33.0",
#     "bitsandbytes>=0.43.0",
#     "openenv-core[core]>=0.2.2",
#     "matplotlib",
#     "numpy",
#     "requests",
#     "huggingface_hub",
# ]
# ///
"""HF Jobs training script for Incident Commander RL env."""

# ---------------------------------------------------------------------
# 1. Persistent caching (set BEFORE any HF/torch import)
# ---------------------------------------------------------------------
import os, sys, subprocess
DATA_DIR = "/data" if os.path.isdir("/data") else "/tmp"
os.environ.setdefault("HF_HOME",            f"{DATA_DIR}/hf")
os.environ.setdefault("TRANSFORMERS_CACHE", f"{DATA_DIR}/hf/transformers")
os.environ.setdefault("HF_HUB_CACHE",       f"{DATA_DIR}/hf/hub")
os.environ.setdefault("PIP_CACHE_DIR",      f"{DATA_DIR}/pip-cache")
os.environ.setdefault("UV_CACHE_DIR",       f"{DATA_DIR}/uv-cache")
for d in (os.environ["HF_HOME"], os.environ["PIP_CACHE_DIR"], os.environ["UV_CACHE_DIR"]):
    os.makedirs(d, exist_ok=True)

# ---------------------------------------------------------------------
# 2. Clone the project repo
# ---------------------------------------------------------------------
REPO_URL = os.environ.get("REPO_URL",
    "https://github.com/Unknown-guy-369/Incident_commander.git")
REPO_DIR = f"{DATA_DIR}/incident_commander"
if not os.path.isdir(REPO_DIR):
    print(f"Cloning {REPO_URL} -> {REPO_DIR}")
    subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)
else:
    print(f"Repo already at {REPO_DIR}; pulling latest")
    subprocess.run(["git", "-C", REPO_DIR, "pull", "--rebase"], check=False)
sys.path.insert(0, REPO_DIR)
os.chdir(REPO_DIR)

# ---------------------------------------------------------------------
# 3. Configuration
# ---------------------------------------------------------------------
MODEL_NAME       = os.environ.get("MODEL_NAME", "unsloth/Llama-3.2-1B-Instruct")
MAX_SEQ_LENGTH   = int(os.environ.get("MAX_SEQ_LENGTH", "2048"))
LORA_RANK        = int(os.environ.get("LORA_RANK", "16"))
BATCH_SIZE       = int(os.environ.get("BATCH_SIZE", "1"))
GRAD_ACCUM_STEPS = int(os.environ.get("GRAD_ACCUM_STEPS", "4"))
LEARNING_RATE    = float(os.environ.get("LEARNING_RATE", "2e-5"))
NUM_GENERATIONS  = int(os.environ.get("NUM_GENERATIONS", "2"))
ROLLOUT_STEPS    = int(os.environ.get("ROLLOUT_STEPS", "8"))
MAX_STEPS        = int(os.environ.get("MAX_STEPS", "30"))
DIFFICULTY       = int(os.environ.get("DIFFICULTY", "1"))
NUM_PROMPTS      = int(os.environ.get("NUM_PROMPTS", "80"))
NUM_EVAL_EPISODES= int(os.environ.get("NUM_EVAL_EPISODES", "15"))
RESUME_FROM      = os.environ.get("RESUME_FROM_CHECKPOINT", "").strip() or None
HF_REPO_ID       = os.environ.get("HF_REPO_ID", "").strip() or None
HF_TOKEN         = os.environ.get("HF_TOKEN", "").strip() or None
HF_SPACE_URL     = os.environ.get(
    "INCIDENT_COMMANDER_ENV_URL",
    "https://abishek-priyan-369-incident-commander.hf.space",
)

OUTPUT_DIR = f"{DATA_DIR}/outputs/commander_grpo"
ASSETS_DIR = f"{DATA_DIR}/outputs/assets"
FINAL_DIR  = f"{DATA_DIR}/outputs/commander_final"
for d in (OUTPUT_DIR, ASSETS_DIR, FINAL_DIR):
    os.makedirs(d, exist_ok=True)

print(f"DATA_DIR    = {DATA_DIR}")
print(f"HF_HOME     = {os.environ['HF_HOME']}")
print(f"REPO_DIR    = {REPO_DIR}")
print(f"OUTPUT_DIR  = {OUTPUT_DIR}")
print(f"MAX_STEPS   = {MAX_STEPS}")
print(f"RESUME_FROM = {RESUME_FROM}")
print(f"HF_REPO_ID  = {HF_REPO_ID}")
print()

# ---------------------------------------------------------------------
# 4. Imports
# ---------------------------------------------------------------------
import torch
import numpy as np
from datasets import Dataset

from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)
from trl import GRPOConfig, GRPOTrainer

from rollout import SyncEnvClient, rollout_episode, SYSTEM_PROMPT
from rewards import make_grpo_reward_fns

print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'}")
print(f"bf16 supported: {torch.cuda.is_bf16_supported()}")
print()

# ---------------------------------------------------------------------
# 5. Load model + LoRA
# ---------------------------------------------------------------------
print(f"Loading {MODEL_NAME}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=torch.float16,
    load_in_4bit=True,
    fast_inference=False,
    max_lora_rank=LORA_RANK,
    gpu_memory_utilization=0.6,
)
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    target_modules=["q_proj","k_proj","v_proj","o_proj",
                    "gate_proj","up_proj","down_proj"],
    lora_alpha=LORA_RANK,
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)
print("Model + LoRA ready.\n")

# ---------------------------------------------------------------------
# 6. Generate fn + episode rollout + reward fns
# ---------------------------------------------------------------------
def make_generate_fn(model, tokenizer, max_new_tokens=160, temperature=0.7):
    device = next(model.parameters()).device
    def _generate(prompt: str) -> str:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                          max_length=1536).to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=temperature > 0, temperature=temperature,
                pad_token_id=tokenizer.eos_token_id, use_cache=True,
            )
        return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                skip_special_tokens=True)
    return _generate

_train_generate_fn = make_generate_fn(model, tokenizer)

def episode_rollout(prompt: str, completion: str):
    with SyncEnvClient(HF_SPACE_URL) as env:
        state = rollout_episode(env, _train_generate_fn,
                                max_steps=ROLLOUT_STEPS, difficulty=DIFFICULTY)
    return state.to_dict()

reward_fns = make_grpo_reward_fns(episode_rollout)
print(f"Wired {len(reward_fns)} reward fns: {[fn.__name__ for fn in reward_fns]}\n")

# ---------------------------------------------------------------------
# 7. Generate prompt dataset
# ---------------------------------------------------------------------
def generate_initial_prompts(num_samples=NUM_PROMPTS, difficulty=DIFFICULTY):
    rows = []
    for i in range(num_samples):
        try:
            with SyncEnvClient(HF_SPACE_URL) as env:
                init = env.reset(difficulty=difficulty)
            obs = init.get("observation", {})
            prompt = (
                f"System: {SYSTEM_PROMPT}\n\n"
                f"Initial alert: {obs.get('alert_summary','')}\n"
                f"Services overview: {obs.get('services_overview','')}\n\n"
                "Begin the investigation. What is your first action?"
            )
            rows.append({"prompt": prompt, "seed": i})
        except Exception as e:
            print(f"  prompt {i} failed: {e}")
        if (i + 1) % 25 == 0:
            print(f"  generated {i+1}/{num_samples}")
    return Dataset.from_list(rows)

print("Generating prompt dataset...")
dataset = generate_initial_prompts()
print(f"Dataset size: {len(dataset)}\n")

# ---------------------------------------------------------------------
# 8. Eval helper
# ---------------------------------------------------------------------
def evaluate_episodes(num_episodes, label, max_steps=ROLLOUT_STEPS,
                     difficulty=DIFFICULTY, temperature=0.7):
    print(f"=== Evaluating: {label} ({num_episodes} episodes) ===")
    FastLanguageModel.for_inference(model)
    gen_fn = make_generate_fn(model, tokenizer, max_new_tokens=160,
                               temperature=temperature)
    rewards, valid_actions = [], []
    resolved_count, correct_root_cause = 0, 0
    per_signal = {"service_recovery":[], "root_cause_accuracy":[],
                  "action_quality":[], "speed":[]}
    for ep in range(num_episodes):
        try:
            with SyncEnvClient(HF_SPACE_URL) as env:
                state = rollout_episode(env, gen_fn, max_steps=max_steps,
                                        difficulty=difficulty)
        except Exception as e:
            print(f"  episode {ep} failed: {e}")
            continue
        bd = (state.last_observation or {}).get("reward_breakdown") or {}
        rewards.append(state.last_reward)
        for k in per_signal:
            per_signal[k].append(float(bd.get(k, 0.0)))
        valid_actions.append(int(any(
            a in {"read_logs","read_metrics","identify_cause"}
            for a in state.actions_taken)))
        if (state.post_fix_status == "recovered"
            and "resolve" in state.actions_taken):
            resolved_count += 1
        if (state.locked_hypothesis
            and state.locked_hypothesis == state.true_root_cause):
            correct_root_cause += 1
        hyp_mark = ("OK" if state.locked_hypothesis == state.true_root_cause
                    else "X")
        print(f"  ep {ep+1:>2}/{num_episodes}  steps={state.steps_used:>2}  "
              f"reward={state.last_reward:.3f}  status={state.post_fix_status}  "
              f"hyp={hyp_mark}")
    FastLanguageModel.for_training(model)
    summary = {
        "label": label,
        "avg_reward": float(np.mean(rewards)) if rewards else 0.0,
        "valid_action_rate": float(np.mean(valid_actions)) if valid_actions else 0.0,
        "resolve_rate": resolved_count / max(num_episodes, 1),
        "root_cause_accuracy": correct_root_cause / max(num_episodes, 1),
        "per_signal_mean": {k: float(np.mean(v)) if v else 0.0
                            for k, v in per_signal.items()},
        "rewards": rewards,
    }
    print(f"  Summary: avg_reward={summary['avg_reward']:.4f}  "
          f"resolve_rate={summary['resolve_rate']*100:.1f}%  "
          f"root_cause_accuracy={summary['root_cause_accuracy']*100:.1f}%\n")
    return summary

# ---------------------------------------------------------------------
# 9. BEFORE eval (skip on resume)
# ---------------------------------------------------------------------
if RESUME_FROM:
    print("RESUME_FROM is set; skipping BEFORE eval.\n")
    before = None
else:
    before = evaluate_episodes(NUM_EVAL_EPISODES, "Before GRPO")

# ---------------------------------------------------------------------
# 10. GRPO training
# ---------------------------------------------------------------------
training_args = GRPOConfig(
    output_dir=OUTPUT_DIR,
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    max_prompt_length=1024,
    max_completion_length=200,
    num_generations=NUM_GENERATIONS,
    max_steps=MAX_STEPS,
    logging_steps=2,
    save_steps=10,
    save_total_limit=5,
    beta=0.04,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    report_to="none",
)
trainer = GRPOTrainer(
    model=model, reward_funcs=reward_fns, args=training_args,
    train_dataset=dataset, processing_class=tokenizer,
)

print(f"Starting GRPO training (max_steps={MAX_STEPS})")
if RESUME_FROM:
    print(f"Resuming from {RESUME_FROM}")
    trainer.train(resume_from_checkpoint=RESUME_FROM)
else:
    trainer.train()
print("Training complete.\n")

# ---------------------------------------------------------------------
# 11. Save merged 16-bit model
# ---------------------------------------------------------------------
print(f"Saving merged 16-bit model to {FINAL_DIR}")
model.save_pretrained_merged(FINAL_DIR, tokenizer, save_method="merged_16bit")

# ---------------------------------------------------------------------
# 12. AFTER eval
# ---------------------------------------------------------------------
after = evaluate_episodes(NUM_EVAL_EPISODES, f"After GRPO ({MAX_STEPS} steps)")

# ---------------------------------------------------------------------
# 13. Plots
# ---------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if before is None:
    before = {
        "avg_reward": 0.0, "resolve_rate": 0.0, "root_cause_accuracy": 0.0,
        "per_signal_mean": {k: 0.0 for k in after["per_signal_mean"]},
        "rewards": [0.0],
    }

metrics_names = ["Avg Reward", "Resolve Rate", "Root-cause Accuracy"]
before_vals = [before["avg_reward"], before["resolve_rate"],
               before["root_cause_accuracy"]]
after_vals  = [after["avg_reward"],  after["resolve_rate"],
               after["root_cause_accuracy"]]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
x = np.arange(len(metrics_names)); w = 0.35
axes[0].bar(x - w/2, before_vals, w, label="Before",
            color="#ff6b6b", alpha=0.85)
axes[0].bar(x + w/2, after_vals,  w, label="After",
            color="#51cf66", alpha=0.85)
axes[0].set_xticks(x); axes[0].set_xticklabels(metrics_names)
axes[0].set_title(f"Before vs After GRPO ({MAX_STEPS} steps)",
                  fontweight="bold")
axes[0].legend(); axes[0].grid(axis="y", alpha=0.3)
axes[1].hist(after["rewards"], bins=10, color="#51cf66",
             alpha=0.7, edgecolor="black")
axes[1].axvline(before["avg_reward"], color="red", linestyle="--",
                linewidth=2, label=f"Before: {before['avg_reward']:.3f}")
axes[1].axvline(after["avg_reward"],  color="green", linestyle="--",
                linewidth=2, label=f"After:  {after['avg_reward']:.3f}")
axes[1].set_title("Post-training reward distribution", fontweight="bold")
axes[1].set_xlabel("Reward"); axes[1].set_ylabel("Episodes")
axes[1].legend(); axes[1].grid(alpha=0.3)
deltas = [after_vals[i] - before_vals[i] for i in range(len(metrics_names))]
colors = ["#51cf66" if d >= 0 else "#ff6b6b" for d in deltas]
axes[2].bar(metrics_names, deltas, color=colors, alpha=0.85,
            edgecolor="black")
axes[2].axhline(0, color="black", linewidth=0.8)
axes[2].set_title("Improvement delta", fontweight="bold")
axes[2].grid(axis="y", alpha=0.3)
plt.tight_layout()
fig.savefig(f"{ASSETS_DIR}/before_after_comparison.png",
            dpi=150, bbox_inches="tight")
plt.close(fig)

fig2, ax2 = plt.subplots(figsize=(10, 5))
signals = list(before["per_signal_mean"].keys())
b_means = [before["per_signal_mean"][k] for k in signals]
a_means = [after["per_signal_mean"][k]  for k in signals]
x = np.arange(len(signals)); w = 0.35
ax2.bar(x - w/2, b_means, w, label="Before", color="#ff6b6b", alpha=0.85)
ax2.bar(x + w/2, a_means, w, label="After",  color="#51cf66", alpha=0.85)
ax2.set_xticks(x); ax2.set_xticklabels(signals, rotation=15, ha="right")
ax2.set_title("Per-signal reward (raw) - before vs after", fontweight="bold")
ax2.legend(); ax2.grid(axis="y", alpha=0.3)
plt.tight_layout()
fig2.savefig(f"{ASSETS_DIR}/per_signal_comparison.png",
             dpi=150, bbox_inches="tight")
plt.close(fig2)

print(f"Plots saved to {ASSETS_DIR}/")
print()
print("=" * 60)
print(f"Incident Commander - GRPO results ({MAX_STEPS} steps)")
print("=" * 60)
print(f"{'Metric':<25}{'Before':>10}{'After':>10}{'Delta':>10}")
print("-" * 60)
for name, b, a in zip(metrics_names, before_vals, after_vals):
    delta = a - b
    arrow = "UP" if delta > 0 else ("DOWN" if delta < 0 else "FLAT")
    print(f"{name:<25}{b:>10.3f}{a:>10.3f}{delta:>+10.3f} {arrow}")
print("=" * 60)

# ---------------------------------------------------------------------
# 14. Push merged model + plots to HF Hub
# ---------------------------------------------------------------------
if HF_REPO_ID and HF_TOKEN:
    print(f"\nPushing to https://huggingface.co/{HF_REPO_ID}")
    try:
        from huggingface_hub import HfApi, login
        login(token=HF_TOKEN)
        api = HfApi()
        api.create_repo(repo_id=HF_REPO_ID, exist_ok=True)
        api.upload_folder(folder_path=FINAL_DIR,  repo_id=HF_REPO_ID,
                          repo_type="model")
        api.upload_folder(folder_path=ASSETS_DIR, repo_id=HF_REPO_ID,
                          repo_type="model", path_in_repo="assets")
        print(f"Pushed to https://huggingface.co/{HF_REPO_ID}")
    except Exception as e:
        print(f"Push failed: {e}")
else:
    print("\nSkipping HF Hub push (HF_REPO_ID and HF_TOKEN must both be set)")

print("\n=== Done ===")