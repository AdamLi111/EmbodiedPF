"""Compare RL-trained policy vs always-execute vs random baselines, all with real API.

Usage:
    python rl_project/scripts/compare_real.py --num_episodes 50

Evaluates three strategies:
  1. RL-trained (hierarchical checkpoint from real API training)
  2. Always-execute (action=0 every turn, like no-friction baseline)
  3. Random (uniform over all 6 actions)
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from rl_project.utils.helpers import load_config, set_seed, get_api_key
from rl_project.envs.friction_env import FrictionDialogueEnv
from rl_project.models.hierarchical_policy import HierarchicalPolicy
from rl_project.models.value_network import ValueNetwork
from rl_project.agents.ppo_agent import PPOAgent
from rl_project.evaluation.metrics import MetricsTracker

API_KEY = get_api_key()


def run_evaluation(env, select_action_fn, num_episodes, label):
    """Run episodes with a given action-selection function."""
    tracker = MetricsTracker()
    t0 = time.time()

    for ep in range(num_episodes):
        obs, info = env.reset()
        state = torch.as_tensor(obs, dtype=torch.float32)
        done = False
        friction_actions = defaultdict(int)
        num_turns = 0

        while not done:
            action = select_action_fn(state)
            obs, _reward, terminated, truncated, info = env.step(action)
            state = torch.as_tensor(obs, dtype=torch.float32)
            done = terminated or truncated
            num_turns += 1

            ft = info.get("friction_type_used")
            if ft:
                friction_actions[ft] += 1

        tracker.add_episode({
            "task_success": info.get("task_success", False),
            "task_failure": info.get("task_failure", False),
            "safety_violation": info.get("safety_violation", False),
            "expected_ambiguity": info.get("expected_ambiguity"),
            "num_turns": num_turns,
            "friction_actions": dict(friction_actions),
        })

        elapsed = time.time() - t0
        rate = (ep + 1) / elapsed
        eta = (num_episodes - ep - 1) / rate if rate > 0 else 0
        print(f"  [{label}] ep {ep+1}/{num_episodes}  "
              f"success={info.get('task_success')}  turns={num_turns}  "
              f"ETA={eta:.0f}s", flush=True)

    return tracker


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/checkpoint_final.pt")
    p.add_argument("--config", default="rl_project/config/default.yaml")
    p.add_argument("--num_episodes", type=int, default=50)
    return p.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(42)

    env = FrictionDialogueEnv(config, openai_api_key=API_KEY, use_encoder=True)
    input_dim = env.obs_dim

    # --- Load RL-trained policy ---
    policy = HierarchicalPolicy(input_dim, config)
    value_net = ValueNetwork(input_dim, config)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    policy.load_state_dict(ckpt["policy_state_dict"])
    value_net.load_state_dict(ckpt["value_state_dict"])
    policy.eval()

    @torch.no_grad()
    def rl_action(state):
        gate_dist, selector_dist = policy.forward(state)
        gate = gate_dist.logits.argmax().item()
        if gate == 0:
            return 0
        return selector_dist.logits.argmax().item() + 1

    def always_execute(state):
        return 0

    def random_action(state):
        return env.action_space.sample()

    strategies = [
        ("RL-trained", rl_action),
        ("Always-execute", always_execute),
        ("Random", random_action),
    ]

    results = {}
    for label, action_fn in strategies:
        print(f"\n{'='*50}")
        print(f"Evaluating: {label} ({args.num_episodes} episodes)")
        print(f"{'='*50}")
        tracker = run_evaluation(env, action_fn, args.num_episodes, label)
        summary = tracker.compute_summary()
        results[label] = summary
        print(f"\n  Result: success={summary['success_rate']:.1%}  "
              f"avg_turns={summary['avg_turns']:.1f}  "
              f"safety={summary['total_safety_violations']}  "
              f"friction={summary['friction_distribution']}")

    # Print comparison table
    print(f"\n{'='*60}")
    print(f"{'Policy':<20} {'Success':>8} {'Avg Turns':>10} {'Safety':>8} ")
    print(f"{'-'*60}")
    for name, s in results.items():
        print(f"{name:<20} {s['success_rate']:>7.1%} {s['avg_turns']:>10.1f} "
              f"{s['total_safety_violations']:>8}")

    # Save
    os.makedirs("rl_project/results", exist_ok=True)
    with open("rl_project/results/comparison_real.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to rl_project/results/comparison_real.json")


if __name__ == "__main__":
    main()
