"""Compare hierarchical, flat, and random-baseline policies.

Usage:
    python -m rl_project.scripts.compare \
        --hierarchical_checkpoint checkpoints/hier_final.pt \
        --flat_checkpoint checkpoints/flat_final.pt \
        --config rl_project/config/default.yaml \
        --num_episodes 500
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import numpy as np
from rl_project.utils.helpers import load_config, set_seed
from rl_project.envs.friction_env import FrictionDialogueEnv
from rl_project.models.flat_policy import FlatPolicy
from rl_project.models.hierarchical_policy import HierarchicalPolicy
from rl_project.models.value_network import ValueNetwork
from rl_project.agents.ppo_agent import PPOAgent
from rl_project.evaluation.evaluator import Evaluator


def _load_agent(policy_cls, checkpoint_path, input_dim, config):
    policy = policy_cls(input_dim, config)
    value_net = ValueNetwork(input_dim, config)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    policy.load_state_dict(ckpt["policy_state_dict"])
    value_net.load_state_dict(ckpt["value_state_dict"])
    policy.eval()
    value_net.eval()
    return PPOAgent(policy, value_net, config)


class RandomAgent:
    """Baseline that samples actions uniformly."""
    def __init__(self, input_dim, config):
        self.policy = _RandomPolicy()
        self.value_network = _RandomValue()
    def select_action(self, state):
        a = np.random.randint(0, 6)
        return a, 0.0, 0.0


class _RandomPolicy:
    def forward(self, state):
        from torch.distributions import Categorical
        return Categorical(logits=torch.zeros(6))
    def parameters(self):
        return iter([])


class _RandomValue:
    def __call__(self, state):
        return torch.tensor(0.0)
    def parameters(self):
        return iter([])


def parse_args():
    p = argparse.ArgumentParser(description="Compare friction policies")
    p.add_argument("--hierarchical_checkpoint", required=True)
    p.add_argument("--flat_checkpoint", required=True)
    p.add_argument("--config", default="rl_project/config/default.yaml")
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

    agents = {
        "hierarchical": _load_agent(HierarchicalPolicy,
                                    args.hierarchical_checkpoint,
                                    input_dim, config),
        "flat": _load_agent(FlatPolicy, args.flat_checkpoint,
                            input_dim, config),
        "random": RandomAgent(input_dim, config),
    }

    results = {}
    for name, agent in agents.items():
        print(f"\nEvaluating {name} policy ({args.num_episodes} episodes)...")
        evaluator = Evaluator(env, agent, config)
        tracker = evaluator.evaluate(args.num_episodes)
        results[name] = tracker.compute_summary()
        print(f"  success={results[name]['success_rate']:.2%}  "
              f"avg_turns={results[name]['avg_turns']:.1f}  "
              f"safety={results[name]['total_safety_violations']}")

    # Save results
    os.makedirs("rl_project/results", exist_ok=True)
    with open("rl_project/results/comparison.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Generate plots
    _plot_comparison(results)
    print("\nResults and plots saved to rl_project/results/")


def _plot_comparison(results):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plots")
        return

    out = "rl_project/results"
    names = list(results.keys())

    # 1. Success rate by ambiguity type (grouped bar)
    all_ambs = sorted({
        a for r in results.values()
        for a in r.get("success_by_ambiguity", {})
    })
    if all_ambs:
        x = np.arange(len(all_ambs))
        width = 0.25
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, name in enumerate(names):
            vals = [results[name].get("success_by_ambiguity", {}).get(a, 0)
                    for a in all_ambs]
            ax.bar(x + i * width, vals, width, label=name)
        ax.set_xticks(x + width)
        ax.set_xticklabels(all_ambs, rotation=30, ha="right")
        ax.set_ylabel("Success rate")
        ax.set_title("Success Rate by Ambiguity Type")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out, "success_by_ambiguity.png"), dpi=150)
        plt.close(fig)

    # 2. Avg turns bar chart
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(names, [results[n]["avg_turns"] for n in names])
    ax.set_ylabel("Average turns")
    ax.set_title("Average Episode Length")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "avg_turns.png"), dpi=150)
    plt.close(fig)

    # 3. Friction distribution pie charts
    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 4))
    if len(names) == 1:
        axes = [axes]
    for ax, name in zip(axes, names):
        fd = results[name].get("friction_distribution", {})
        if fd:
            ax.pie(fd.values(), labels=fd.keys(), autopct="%1.0f%%")
        ax.set_title(name)
    fig.suptitle("Friction Type Distribution")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "friction_distribution.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
