"""
AI Incident Commander — Demo Inference Script
Loads a trained Unsloth model and runs a full episode in the environment.
"""

import os
import time
import argparse
from unsloth import FastLanguageModel

from server.incident_commander_environment import IncidentCommanderEnvironment
from client import IncidentCommanderEnv
from models import IncidentCommanderAction

import re

REMOTE_ENV_URL = os.environ.get(
    "INCIDENT_COMMANDER_ENV_URL",
    "https://abishek-priyan-369-incident-commander.hf.space",
).rstrip("/")

# Format tag parsing
def parse_action(completion: str):
    match = re.search(r"<action>(.*?):?(.*?)</action>", completion)
    if match:
        atype = match.group(1).strip()
        target = match.group(2).strip() if match.group(2) else "payment-service"
        return atype, target
    return None, None

def run_episode(model, tokenizer, difficulty=1, max_steps=50, env_url=None):
    print(f"\n--- Starting Evaluation Episode (Difficulty {difficulty}) ---")
    use_remote_env = bool(env_url)

    if use_remote_env:
        print(f"Using remote environment: {env_url}")
        env = IncidentCommanderEnv(base_url=env_url)
    else:
        print("Using local in-process environment.")
        env = IncidentCommanderEnvironment(difficulty=difficulty)

    obs = env.reset()
    
    system_prompt = """You are an AI Incident Commander.
You must investigate microservice outages by using tools and reasoning causally.
Output your reasoning in <thought> tags, then lock your hypothesis and apply a fix.
Action format: <action>action_name:target_service</action>"""

    total_reward = 0
    
    for step in range(max_steps):
        print(f"\n[Step {step + 1}/{max_steps}]")
        
        # Format the observation for the model
        prompt = f"System: {system_prompt}\n\nObservation: {obs.alert_summary}\n"
        if obs.last_action_result:
            prompt += f"Previous Action Result: {obs.last_action_result}\n"
        prompt += f"Services Overview: {obs.services_overview}\n\n"
        prompt += "What is your next action? Think step by step."
        
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=200, use_cache=True)
        
        # Decode and parse
        completion = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
        print(f"Agent Output:\n{completion}\n")
        
        action_type, target = parse_action(completion)
        if not action_type:
            print("❌ Agent produced an invalid format! Ending episode.")
            break
            
        print(f"Action parsed: {action_type} on {target}")
        
        # Step the environment
        step_result = env.step(
            IncidentCommanderAction(action_type=action_type, target_service=target)
        )
        obs = step_result.observation if use_remote_env else step_result
        total_reward = float(step_result.reward)
        
        if obs.done:
            print(f"\n✅ Episode finished! Final result: {obs.last_action_result}")
            print(f"🏆 Final Reward: {total_reward:.3f}")
            if obs.reward_breakdown:
                print(f"📊 Breakdown: {obs.reward_breakdown}")
            break
            
    if not obs.done:
        print("\n❌ Episode timed out!")
    if use_remote_env:
        env.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="outputs/commander_final", help="Path to trained model")
    parser.add_argument("--difficulty", type=int, default=1, choices=[1, 2, 3, 4], help="Scenario difficulty")
    parser.add_argument(
        "--env-url",
        type=str,
        default=REMOTE_ENV_URL,
        help="Remote OpenEnv base URL. Empty string uses local environment.",
    )
    args = parser.parse_args()
    
    print(f"Loading model from {args.model}...")
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model,
            max_seq_length=2048,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)
        selected_env_url = (args.env_url or "").strip()
        run_episode(
            model,
            tokenizer,
            difficulty=args.difficulty,
            env_url=selected_env_url if selected_env_url else None,
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Note: If the model is not trained yet, run 'python training.py --run' first.")

if __name__ == "__main__":
    main()
