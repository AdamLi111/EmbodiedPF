"""Train hierarchical policy with real API calls + encoder for 500 steps."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from rl_project.utils.helpers import load_config, set_seed, get_api_key
from rl_project.envs.friction_env import FrictionDialogueEnv
from rl_project.models.hierarchical_policy import HierarchicalPolicy
from rl_project.models.value_network import ValueNetwork
from rl_project.agents.ppo_agent import PPOAgent
from rl_project.training.reward import RewardComputer
from rl_project.training.trainer import Trainer
from rl_project.utils.logger import Logger

config = load_config("rl_project/config/default.yaml")
config["reward"]["per_turn_penalty"] = -0.15
config["ppo"]["total_timesteps"] = 10000
config["ppo"]["rollout_steps"] = 50
config["training"]["log_interval"] = 1
config["training"]["eval_interval"] = 9999  # skip eval to save API calls

set_seed(42)

env = FrictionDialogueEnv(
    config,
    openai_api_key=get_api_key(),
    use_encoder=True,
)
policy = HierarchicalPolicy(env.obs_dim, config)
value_net = ValueNetwork(env.obs_dim, config)
agent = PPOAgent(policy, value_net, config)
trainer = Trainer(env, agent, RewardComputer(config), config, Logger("runs_full_500/"))
trainer.train()
