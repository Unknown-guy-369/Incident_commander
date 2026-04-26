"""
AI Incident Commander — local inference / evaluation script.

Loads a trained Unsloth model and runs full multi-turn episodes against
the live environment (remote HF Space by default; pass --env-url '' for
the in-process environment).

Usage:
    python inference.py --model outputs/commander_final --difficulty 1
    python inference.py --model outputs/commander_final --episodes 20
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys

# Make sibling imports work whether we're a script or a package member.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rollout import (  # noqa: E402
    DEFAULT_REMOTE_URL,
    SYSTEM_PROMPT,
    SyncEnvClient,
    LocalEnvAdapter,
    rollout_episode,
)


def _make_local_env(difficulty: int):
    try:
        from server.incident_commander_environment import IncidentCommanderEnvironment
    except (ImportError, ModuleNotFoundError):
        from incident_commander.server.incident_commander_environment import (  # type: ignore
            IncidentCommanderEnvironment,
        )
    return LocalEnvAdapter(IncidentCommanderEnvironment(difficulty=difficulty), difficulty=difficulty)


def make_generate_fn(model, tokenizer, max_new_tokens: int = 200, temperature: float = 0.7):
    import torch

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
        return tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

    return _generate


def _on_step(step_idx: int, completion: str, obs: dict) -> None:
    print(f"\n[Step {step_idx + 1}]")
    print(f"  Agent: {completion[:200].strip()}")
    if obs.get("last_action_result"):
        print(f"  Result: {obs['last_action_result']}")


def run_episode(model, tokenizer, *, difficulty: int = 1, max_steps: int = 25,
                env_url: str = "", verbose: bool = True) -> dict:
    """Run one episode against either a remote OpenEnv server or the local
    in-process env.

    env_url:
        Empty string  -> use local in-process env (recommended; matches
                         the demo's setup, no HTTP session bugs).
        Any URL       -> SyncEnvClient against that URL (e.g. the live HF Space).
    """
    use_local = not (env_url and env_url.strip())
    if use_local:
        env_ctx = _make_local_env(difficulty)
        if verbose:
            print("Using local in-process env (no HTTP)")
    else:
        env_ctx = SyncEnvClient(env_url)
        if verbose:
            print(f"Using remote env: {env_url}")

    generate_fn = make_generate_fn(model, tokenizer)

    with env_ctx as env:
        state = rollout_episode(
            env, generate_fn,
            max_steps=max_steps,
            difficulty=difficulty,
            on_step=_on_step if verbose else None,
        )

    bd = (state.last_observation or {}).get("reward_breakdown") or {}
    summary = {
        "actions_taken": state.actions_taken,
        "steps_used": state.steps_used,
        "post_fix_status": state.post_fix_status,
        "locked_hypothesis": state.locked_hypothesis,
        "true_root_cause": state.true_root_cause or bd.get("true_root_cause"),
        "reward": state.last_reward,
        "reward_breakdown": bd,
        "resolved": state.post_fix_status == "recovered" and "resolve" in state.actions_taken,
    }

    if verbose:
        print("\n--- Episode summary ---")
        for k in ("steps_used", "post_fix_status", "locked_hypothesis", "true_root_cause", "reward"):
            print(f"  {k}: {summary[k]}")
        if bd:
            print(f"  breakdown: {bd}")

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="outputs/commander_final",
                        help="Path to trained model")
    parser.add_argument("--difficulty", type=int, default=1, choices=[1, 2, 3, 4])
    parser.add_argument("--max-steps", type=int, default=25,
                        help="Max env steps per episode")
    parser.add_argument("--episodes", type=int, default=10,
                        help="How many episodes to evaluate (default 10)")
    parser.add_argument(
        "--env-url",
        type=str,
        default="",
        help="Remote OpenEnv base URL. Empty string = use local in-process env "
             "(recommended). Set to e.g. https://...hf.space to use a deployed Space.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress per-step output")
    args = parser.parse_args()

    print(f"Loading model from {args.model}...")
    try:
        from unsloth import FastLanguageModel  # type: ignore
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model,
            max_seq_length=2048,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)
    except Exception as e:  # noqa: BLE001
        print(f"Error loading model: {e}")
        print("If the model isn't trained yet, train it first or pull from HF Hub.")
        return

    rewards: list = []
    resolved = 0
    for ep in range(args.episodes):
        if args.episodes > 1:
            print(f"\n{'#' * 60}\n# Episode {ep + 1}/{args.episodes}\n{'#' * 60}")
        summary = run_episode(
            model, tokenizer,
            difficulty=args.difficulty,
            max_steps=args.max_steps,
            env_url=args.env_url,
            verbose=not args.quiet,
        )
        rewards.append(summary["reward"])
        resolved += int(summary["resolved"])

    print(f"\n{'=' * 60}")
    print(f"Results over {args.episodes} episode(s) (difficulty {args.difficulty})")
    print(f"  Resolved:    {resolved}/{args.episodes} ({100 * resolved / max(args.episodes, 1):.1f}%)")
    print(f"  Mean reward: {statistics.mean(rewards):.4f}")
    if len(rewards) > 1:
        print(f"  Std  reward: {statistics.pstdev(rewards):.4f}")
        print(f"  Min  reward: {min(rewards):.4f}")
        print(f"  Max  reward: {max(rewards):.4f}")


if __name__ == "__main__":
    main()
