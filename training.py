"""
AI Incident Commander — GRPO Training Pipeline
Uses unsloth for memory efficiency and trl's GRPOTrainer for RL training.
"""

import os
import torch
from datasets import Dataset
from unsloth import FastLanguageModel, PatchFastRL
import re

# Patch Unsloth for GRPO support
PatchFastRL("GRPO", FastLanguageModel)
from trl import GRPOConfig, GRPOTrainer

# Import our environment
from server.incident_commander_environment import IncidentCommanderEnvironment
from models import IncidentCommanderAction

# --- 1. Configuration ---
MODEL_NAME = "unsloth/Llama-3.2-1B-Instruct"  # Lightweight model for training
MAX_SEQ_LENGTH = 2048
LORA_RANK = 16
BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 4
LEARNING_RATE = 2e-5

# --- 2. Prompts & Dataset ---
SYSTEM_PROMPT = """You are an AI Incident Commander.
You must investigate microservice outages by using tools and reasoning causally.
Output your reasoning in <thought> tags, then lock your hypothesis and apply a fix.
Action format: <action>action_name:target_service</action>"""

def generate_initial_prompts(num_samples=100, difficulty=1):
    """Generate initial environment states for GRPO."""
    prompts = []
    
    for _ in range(num_samples):
        env = IncidentCommanderEnvironment(difficulty=difficulty)
        obs = env.reset()
        
        # Format the starting observation
        prompt = f"System: {SYSTEM_PROMPT}\n\nObservation: {obs.alert_summary}\n"
        prompt += f"Services Overview: {obs.services_overview}\n\n"
        prompt += "What is your next action?"
        
        prompts.append({
            "prompt": prompt,
            "seed": env._ctx.scenario.seed if env._ctx else 0
        })
        
    return Dataset.from_list(prompts)

# --- 3. Reward Functions ---
def parse_action(completion: str):
    """Simple parser for <action>type:target</action>."""
    match = re.search(r"<action>\s*([^:<]+)(?::([^<]+))?\s*</action>", completion)
    if match:
        atype = match.group(1).strip()
        target = match.group(2).strip() if match.group(2) else "payment-service"
        return atype, target
    return None, None

def environment_reward_func(prompts, completions, **kwargs):
    """
    Evaluates the model's completion using the OpenEnv environment.
    GRPOs generate completions; we run them through the environment to calculate the reward.
    """
    rewards = []
    for prompt, completion in zip(prompts, completions):
        # We can extract the seed from kwargs if we passed it, or just use difficulty 1
        env = IncidentCommanderEnvironment(difficulty=1)
        env.reset()
        
        # Extract the action from the completion
        action_type, target = parse_action(completion)
        
        if not action_type:
            # Failed to follow generation format
            rewards.append(0.0)
            continue
            
        # Execute the step in the environment
        try:
            obs = env.step(IncidentCommanderAction(action_type=action_type, target_service=target))
            # Env returns reward in [0, 1] range; GRPOTrainer handles positive/critical rewards
            rewards.append(obs.reward)
        except Exception as e:
            print(f"Env error: {e}")
            rewards.append(0.0)
            
    return rewards

def format_reward_func(prompts, completions, **kwargs):
    """Bonus reward for correctly using <thought> and <action> tags."""
    rewards = []
    for completion in completions:
        score = 0.0
        if "<thought>" in completion and "</thought>" in completion:
            score += 0.1
        if "<action>" in completion and "</action>" in completion:
            score += 0.1
        rewards.append(score)
    return rewards

# --- 4. Main Training Loop ---
def main():
    print("Loading lightweight model via Unsloth...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        fast_inference=True, # Enable vLLM speedups for GRPO
        max_lora_rank=LORA_RANK,
        gpu_memory_utilization=0.6, 
    )
    
    # Configure PEFT
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=LORA_RANK,
        use_gradient_checkpointing="unsloth", 
        random_state=3407,
    )
    
    print("Generating curriculum tasks...")
    # Start on Level 1 (Easy) as per the build plan
    dataset = generate_initial_prompts(num_samples=250, difficulty=1)
    
    print("Initializing GRPOTrainer...")
    training_args = GRPOConfig(
        output_dir="outputs/commander_grpo",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        max_prompt_length=1024,
        max_completion_length=512,
        num_train_epochs=1,
        logging_steps=10,
        beta=0.1,  # KL penalty
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
    )
    
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[environment_reward_func, format_reward_func],
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    
    print("Starting Phase 3 Training (GRPO + Unsloth)...")
    trainer.train()
    
    print("Saving trained model...")
    model.save_pretrained_merged("outputs/commander_final", tokenizer, save_method="merged_16bit")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="Run the training pipeline")
    args = parser.parse_args()
    
    if args.run:
        main()
    else:
        print("Training script written. Run with 'python training.py --run'")
