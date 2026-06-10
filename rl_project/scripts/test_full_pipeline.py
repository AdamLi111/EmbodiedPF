"""Quick 10-step test of the full pipeline with real API + encoder."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from rl_project.utils.helpers import load_config, get_api_key
from rl_project.envs.friction_env import FrictionDialogueEnv
from rl_project.models.hierarchical_policy import HierarchicalPolicy
from rl_project.models.value_network import ValueNetwork
from rl_project.agents.ppo_agent import PPOAgent
from rl_project.training.reward import RewardComputer
from rl_project.training.trainer import Trainer
from rl_project.utils.logger import Logger

config = load_config("rl_project/config/default.yaml")
config["ppo"]["total_timesteps"] = 10
config["ppo"]["rollout_steps"] = 10
config["training"]["log_interval"] = 1

env = FrictionDialogueEnv(
    config,
    openai_api_key=get_api_key(),
    use_encoder=True,
)
policy = HierarchicalPolicy(env.obs_dim, config)
value_net = ValueNetwork(env.obs_dim, config)
agent = PPOAgent(policy, value_net, config)
trainer = Trainer(env, agent, RewardComputer(config), config, Logger("runs_full_test/"))
trainer.train()
