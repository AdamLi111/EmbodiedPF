"""Train both hierarchical and flat policies for 500k steps, then compare."""

import os
import sys
import shutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from rl_project.utils.helpers import load_config, set_seed
from rl_project.envs.friction_env import FrictionDialogueEnv
from rl_project.models.hierarchical_policy import HierarchicalPolicy
from rl_project.models.flat_policy import FlatPolicy
from rl_project.models.value_network import ValueNetwork
from rl_project.agents.ppo_agent import PPOAgent
from rl_project.training.reward import RewardComputer
from rl_project.training.trainer import Trainer
from rl_project.utils.logger import Logger

config = load_config("rl_project/config/default.yaml")
config["reward"]["per_turn_penalty"] = -0.15
config["ppo"]["total_timesteps"] = 500000

os.makedirs("checkpoints", exist_ok=True)

# --- Hierarchical ---
print("=" * 60)
print("Training HIERARCHICAL policy (500k steps)")
print("=" * 60)
set_seed(42)
env1 = FrictionDialogueEnv(config, openai_api_key="test")
Trainer(
    env1,
    PPOAgent(HierarchicalPolicy(env1.obs_dim, config),
             ValueNetwork(env1.obs_dim, config), config),
    RewardComputer(config), config, Logger("runs/hier"),
).train()
shutil.copy("checkpoints/checkpoint_final.pt", "checkpoints/hier_final.pt")

# --- Flat ---
print("\n" + "=" * 60)
print("Training FLAT policy (500k steps)")
print("=" * 60)
set_seed(42)
env2 = FrictionDialogueEnv(config, openai_api_key="test")
Trainer(
    env2,
    PPOAgent(FlatPolicy(env2.obs_dim, config),
             ValueNetwork(env2.obs_dim, config), config),
    RewardComputer(config), config, Logger("runs/flat"),
).train()
shutil.copy("checkpoints/checkpoint_final.pt", "checkpoints/flat_final.pt")

print("\n" + "=" * 60)
print("Done. Now run comparison:")
print("  KMP_DUPLICATE_LIB_OK=TRUE python -m rl_project.scripts.compare \\")
print("      --hierarchical_checkpoint checkpoints/hier_final.pt \\")
print("      --flat_checkpoint checkpoints/flat_final.pt \\")
print("      --num_episodes 500")
