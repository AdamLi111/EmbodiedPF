"""Entry point: evaluate a trained checkpoint.

Usage:
    python -m rl_project.scripts.evaluate \
        --config rl_project/config/default.yaml \
        --checkpoint checkpoints/checkpoint_final.pt \
        --policy hierarchical \
        --num_episodes 500
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from rl_project.utils.helpers import load_config, set_seed
from rl_project.envs.friction_env import FrictionDialogueEnv
from rl_project.models.flat_policy import FlatPolicy
from rl_project.models.hierarchical_policy import HierarchicalPolicy
from rl_project.models.value_network import ValueNetwork
from rl_project.agents.ppo_agent import PPOAgent
from rl_project.evaluation.evaluator import Evaluator


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate trained friction policy")
    p.add_argument("--config", default="rl_project/config/default.yaml")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--policy", choices=["flat", "hierarchical"],
                   default="hierarchical")
    p.add_argument("--num_episodes", type=int, default=500)
    p.add_argument("--openai_api_key", default="test")
    p.add_argument("--use_encoder", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(config.get("seed", 42))

    env = FrictionDialogueEnv(
        config, openai_api_key=args.openai_api_key,
        use_encoder=args.use_encoder,
    )

    input_dim = env.obs_dim
    if args.policy == "flat":
        policy = FlatPolicy(input_dim, config)
    else:
        policy = HierarchicalPolicy(input_dim, config)
    value_net = ValueNetwork(input_dim, config)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    policy.load_state_dict(ckpt["policy_state_dict"])
    value_net.load_state_dict(ckpt["value_state_dict"])
    policy.eval()
    value_net.eval()

    agent = PPOAgent(policy, value_net, config)
    evaluator = Evaluator(env, agent, config)
    tracker = evaluator.evaluate(args.num_episodes)
    summary = tracker.compute_summary()

    print("\n===== Evaluation Summary =====")
    print(json.dumps(summary, indent=2, default=str))

    # Save results
    os.makedirs("rl_project/results", exist_ok=True)
    out_path = "rl_project/results/eval_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
