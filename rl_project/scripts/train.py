"""Entry point: train an RL friction policy.

Usage:
    python -m rl_project.scripts.train --config rl_project/config/default.yaml --policy hierarchical
"""

import argparse
import os
import sys

# Ensure project root is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from rl_project.utils.helpers import load_config, set_seed
from rl_project.utils.logger import Logger
from rl_project.envs.friction_env import FrictionDialogueEnv
from rl_project.models.flat_policy import FlatPolicy
from rl_project.models.hierarchical_policy import HierarchicalPolicy
from rl_project.models.value_network import ValueNetwork
from rl_project.agents.ppo_agent import PPOAgent
from rl_project.training.reward import RewardComputer
from rl_project.training.trainer import Trainer


def parse_args():
    p = argparse.ArgumentParser(description="Train RL friction policy")
    p.add_argument("--config", default="rl_project/config/default.yaml")
    p.add_argument("--policy", choices=["flat", "hierarchical"],
                   default="hierarchical")
    p.add_argument("--openai_api_key", default="test",
                   help="OpenAI key for LLM prompts ('test' for rule-based mode)")
    p.add_argument("--use_encoder", action="store_true",
                   help="Load sentence-transformer encoder (slower, real embeddings)")
    p.add_argument("--total_timesteps", type=int, default=None,
                   help="Override config total_timesteps (handy for testing)")
    p.add_argument("--log_dir", default="runs/")
    return p.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    if args.total_timesteps is not None:
        config.setdefault("ppo", {})["total_timesteps"] = args.total_timesteps

    set_seed(config.get("seed", 42))

    # Environment
    env = FrictionDialogueEnv(
        config,
        openai_api_key=args.openai_api_key,
        use_encoder=args.use_encoder,
    )

    # Networks
    input_dim = env.obs_dim
    if args.policy == "flat":
        policy = FlatPolicy(input_dim, config)
    else:
        policy = HierarchicalPolicy(input_dim, config)
    value_net = ValueNetwork(input_dim, config)

    print(f"Policy: {args.policy}  |  Input dim: {input_dim}  |  "
          f"Timesteps: {config['ppo']['total_timesteps']}")

    # Agent, reward, logger, trainer
    agent = PPOAgent(policy, value_net, config)
    rc = RewardComputer(config)
    logger = Logger(log_dir=args.log_dir)
    trainer = Trainer(env, agent, rc, config, logger)

    trainer.train()
    logger.close()


if __name__ == "__main__":
    main()
