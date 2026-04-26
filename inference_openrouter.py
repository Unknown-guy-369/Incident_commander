"""
AI Incident Commander — Local Model Inference Script

Runs an episode using a local LLM from Hugging Face via transformers or llama-cpp.
Supports both regular transformer models and GGUF quantized models.

Usage:
    python inference_openrouter.py --model unsloth/Qwen3.6-27B-GGUF --difficulty 1
    python inference_openrouter.py --model meta-llama/Llama-2-7b-hf --device cuda --difficulty 2
"""

import os
import re
import argparse
from dataclasses import dataclass
from typing import Optional
from huggingface_hub import InferenceClient

from server.incident_commander_environment import IncidentCommanderEnvironment
from models import IncidentCommanderAction


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_MODEL = os.environ.get("LOCAL_MODEL", "google/gemma-4-31B-it:novita")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

SYSTEM_PROMPT = """You are an elite AI Incident Commander specialising in SRE.

Your job: investigate microservice outages, identify root causes, and apply fixes.

Episode flow:
  1. Read logs / metrics to investigate
  2. Read deployment history and dependency graph for context
  3. Call identify_cause with your hypothesis
  4. Apply the correct fix (restart_pod | rollback | scale_up | hotfix)
  5. Call monitor_recovery to observe outcome
  6. Call resolve once service is healthy

Rules:
  - You MUST read at least 2 log/metric sources before applying any fix
  - You MUST call identify_cause and lock a root cause hypothesis before fixing
  - Output your reasoning in <thought> tags before every action
  - Format actions as: <action>action_type:target_service</action>
  - If the fix doesn't work (degraded/worse), revise your hypothesis and try again

Root cause → correct fix reference:
  memory_limit_too_low      → scale_up
  bad_deployment            → rollback
  connection_pool_exhausted → hotfix
  traffic_spike             → scale_up
  dependency_failure       → restart_pod
  config_error              → hotfix
  redis_down                → restart_pod
  certificate_expired      → hotfix
"""


# ---------------------------------------------------------------------------
# Action Parser
# ---------------------------------------------------------------------------

def parse_action(completion: str) -> tuple[Optional[str], Optional[str]]:
    """Extract action_type and target_service from <action> tags.

    Tolerant to whitespace, newlines, and the colon being optional (some
    actions like resolve/escalate don't take a target).
    """
    match = re.search(
        r"<action>\s*([^:<\n]+?)\s*(?::\s*([^<\n]+?))?\s*</action>",
        completion,
        re.DOTALL,
    )
    if match:
        action_type = match.group(1).strip()
        target = match.group(2).strip() if match.group(2) else ""
        return action_type, target or None
    return None, None


def parse_hypothesis(completion: str) -> Optional[str]:
    """Pulls a root-cause name out of <hypothesis>...</hypothesis> if present,
    otherwise scans for any of the 8 valid hypothesis tokens."""
    valid = (
        "memory_limit_too_low", "bad_deployment", "connection_pool_exhausted",
        "traffic_spike", "dependency_failure", "config_error",
        "redis_down", "certificate_expired",
    )
    m = re.search(r"<hypothesis>\s*([^<\n]+?)\s*</hypothesis>", completion, re.DOTALL)
    if m:
        candidate = m.group(1).strip()
        return candidate
    for h in valid:
        if h in completion:
            return h
    return None


# ---------------------------------------------------------------------------
# LLM Client
# ---------------------------------------------------------------------------

@dataclass
class LLMResponse:
    content: str
    reasoning: Optional[str] = None
    model: Optional[str] = None
    usage: Optional[dict] = None


class LocalModelLLM:
    """HuggingFace InferenceClient wrapper for Incident Commander."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str = HF_TOKEN,
        temperature: float = 0.3,
        max_tokens: int = 512,
    ):
        if not api_key:
            raise ValueError(
                "HF_TOKEN required. Set HF_TOKEN env var or pass --api-key. "
                "Get your token at https://huggingface.co/settings/tokens"
            )
        
        self.model_name = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        print(f"Using model: {model}")
        self.client = InferenceClient(api_key=api_key)

    def generate(self, prompt: str, system: str = SYSTEM_PROMPT) -> LLMResponse:
        """Generate using HuggingFace Inference API."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        
        content = response.choices[0].message.content
        
        # Extract reasoning from <thought> tags if present
        thought_match = re.search(r"<thought>(.*?)</thought>", content, re.DOTALL)
        reasoning = thought_match.group(1).strip() if thought_match else None
        
        return LLMResponse(
            content=content,
            reasoning=reasoning,
            model=self.model_name,
            usage={
                "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                "completion_tokens": getattr(response.usage, "completion_tokens", 0),
                "total_tokens": getattr(response.usage, "total_tokens", 0),
            },
        )


# ---------------------------------------------------------------------------
# Episode Runner
# ---------------------------------------------------------------------------

def format_observation(obs) -> str:
    """Format the current observation as a prompt suffix for the LLM."""
    lines = [f"Alert: {obs.alert_summary}"]
    lines.append(f"Services: {[s['name'] for s in obs.services_overview]}")

    if obs.last_action_result:
        lines.append(f"Last result: {obs.last_action_result}")

    if obs.steps_remaining < 50:
        lines.append(f"Steps remaining: {obs.steps_remaining}")

    # Show what's been revealed
    if obs.revealed_logs:
        lines.append(f"Logs read: {list(obs.revealed_logs.keys())}")
    if obs.revealed_metrics:
        lines.append(f"Metrics read: {list(obs.revealed_metrics.keys())}")

    # Show locked hypothesis
    if obs.hypothesis_locked:
        lines.append(f"HYPOTHESIS LOCKED: {obs.locked_hypothesis}")

    # Show post-fix status
    if obs.post_fix_status:
        status_emoji = {
            "recovered": "✅",
            "degraded": "⚠️",
            "worse": "🔴",
        }.get(obs.post_fix_status, "❓")
        lines.append(f"Post-fix status: {status_emoji} {obs.post_fix_status}")

    return "\n".join(lines)


def run_episode(
    llm: LocalModelLLM,
    difficulty: int = 1,
    max_steps: int = 50,
    verbose: bool = True,
) -> dict:
    """Run a single episode with a local LLM."""
    env = IncidentCommanderEnvironment(difficulty=difficulty)
    obs = env.reset()

    total_reward = 0.0
    steps = []
    parse_failures = 0

    NO_TARGET_ACTIONS = {
        "read_deployment_history", "read_dependency_graph",
        "monitor_recovery", "escalate", "resolve",
    }

    for step in range(max_steps):
        if verbose:
            print(f"\n{'='*60}")
            print(f"Step {step + 1}/{max_steps} | Steps remaining: {obs.steps_remaining}")

        # Build the prompt
        obs_text = format_observation(obs)
        prompt = (
            f"Current observation:\n{obs_text}\n\n"
            "What is your next action? "
            "Output reasoning in <thought> then action in <action> tags. "
            "For identify_cause also include <hypothesis>name</hypothesis>."
        )

        # Query LLM
        response = llm.generate(prompt)
        if verbose:
            print(f"Model: {response.model} | Tokens: {response.usage['total_tokens']}")
            print(f"\nLLM Output:\n{response.content[:500]}")

        # Parse action — DO NOT inject a default action on parse failure.
        # We track it honestly so the eval reflects model behaviour.
        action_type, target = parse_action(response.content)
        if not action_type:
            parse_failures += 1
            if verbose:
                print("⚠️  No valid <action> tag found. Counting as a wasted step.")
            steps.append({"step": step + 1, "action": "__parse_error__", "target": None})
            continue

        # Only inject a default target for actions that legitimately need one
        # AND the model failed to provide one.
        if not target and action_type not in NO_TARGET_ACTIONS:
            if verbose:
                print(f"⚠️  Missing target for {action_type}; falling back to root service heuristic.")
            target = (obs.services_overview[0]["name"] if obs.services_overview else "payment-service")

        if verbose:
            print(f"→ Action: {action_type} | Target: {target}")

        # Execute in environment — pass hypothesis if applicable.
        try:
            kwargs: dict = {}
            if action_type == "identify_cause":
                hypothesis = parse_hypothesis(response.content)
                if hypothesis:
                    kwargs["hypothesis"] = hypothesis
            if action_type in {"escalate", "resolve"}:
                kwargs["justification"] = "agent decision"
            obs = env.step(IncidentCommanderAction(
                action_type=action_type,
                target_service=target,
                **kwargs,
            ))
        except Exception as e:
            if verbose:
                print(f"⚠️  Env error: {e}")
            steps.append({"step": step + 1, "action": action_type, "target": target, "error": str(e)})
            continue

        total_reward = obs.reward
        steps.append({
            "step": step + 1,
            "action": action_type,
            "target": target,
            "reward": obs.reward,
            "post_fix": obs.post_fix_status,
        })

        if verbose:
            print(f"Reward: {obs.reward:.3f} | Done: {obs.done}")
            if obs.last_action_result:
                print(f"Result: {obs.last_action_result}")

        if obs.done:
            if verbose:
                print(f"\n🏁 Episode ended. {obs.last_action_result}")
            break

    if not obs.done and verbose:
        print(f"\n⏰ Episode timed out after {max_steps} steps.")

    return {
        "done": obs.done,
        "total_reward": total_reward,
        "steps": steps,
        "parse_failures": parse_failures,
        "reward_breakdown": obs.reward_breakdown,
        "final_result": obs.last_action_result,
    }


def run_multi_episode(
    llm: LocalModelLLM,
    num_episodes: int = 5,
    difficulty: int = 1,
    max_steps: int = 50,
) -> list[dict]:
    """Run multiple episodes and aggregate results."""
    results = []
    for i in range(num_episodes):
        print(f"\n{'#'*70}")
        print(f"# Episode {i + 1}/{num_episodes}  (Difficulty {difficulty})")
        print(f"{'#'*70}")
        result = run_episode(llm, difficulty=difficulty, max_steps=max_steps)
        results.append(result)

    # Summary
    print(f"\n{'='*70}")
    print(f"RESULTS OVER {num_episodes} EPISODES (Difficulty {difficulty})")
    print(f"{'='*70}")
    total_rewards = [r["total_reward"] for r in results]
    wins = sum(1 for r in results if r["done"])
    for i, r in enumerate(results):
        status = "✅ RESOLVED" if r["done"] else "❌ TIMEOUT"
        print(f"  Episode {i+1}: reward={r['total_reward']:.3f} | {status}")

    print(f"\n  Win rate:  {wins}/{num_episodes} ({100*wins/num_episodes:.1f}%)")
    print(f"  Mean reward: {sum(total_rewards)/len(total_rewards):.3f}")
    print(f"  Min reward:  {min(total_rewards):.3f}")
    print(f"  Max reward:  {max(total_rewards):.3f}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="HuggingFace Inference API for Incident Commander")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model name from HuggingFace (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=HF_TOKEN,
        help="HuggingFace API key (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--difficulty",
        type=int,
        default=1,
        choices=[1, 2, 3, 4],
        help="Scenario difficulty (1=easy, 4=expert)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of episodes to run (default: 1)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=50,
        help="Max steps per episode (default: 50)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-step LLM output (show only summary)",
    )
    args = parser.parse_args()

    # Validate API key
    api_key = args.api_key or os.environ.get("HF_TOKEN", "")
    if not api_key:
        parser.error(
            "HF_TOKEN required. Pass --api-key or set HF_TOKEN env var.\n"
            "Get your token at https://huggingface.co/settings/tokens"
        )

    print(f"Using model: {args.model}")

    llm = LocalModelLLM(
        model=args.model,
        api_key=api_key,
    )

    if args.episodes == 1:
        result = run_episode(
            llm,
            difficulty=args.difficulty,
            max_steps=args.max_steps,
            verbose=not args.quiet,
        )
        print(f"\nFinal result: {result['final_result']}")
        print(f"Total reward: {result['total_reward']:.3f}")
    else:
        run_multi_episode(
            llm,
            num_episodes=args.episodes,
            difficulty=args.difficulty,
            max_steps=args.max_steps,
        )


if __name__ == "__main__":
    main()
