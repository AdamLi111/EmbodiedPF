"""Run a trained policy greedily and collect episode-level metrics."""

import torch
import numpy as np
from collections import defaultdict

from .metrics import MetricsTracker


class Evaluator:
    def __init__(self, env, agent, config):
        self.env = env
        self.agent = agent
        self.config = config

    @torch.no_grad()
    def evaluate(self, num_episodes: int) -> MetricsTracker:
        tracker = MetricsTracker()

        for _ in range(num_episodes):
            obs, info = self.env.reset()
            state = torch.as_tensor(obs, dtype=torch.float32)
            done = False
            friction_actions = defaultdict(int)
            num_turns = 0

            while not done:
                # Greedy: pick argmax from policy distribution
                dist = self.agent.policy.forward(state)
                if hasattr(dist, 'logits'):
                    # Flat policy returns single Categorical
                    action = dist.logits.argmax().item()
                else:
                    # Hierarchical returns (gate_dist, selector_dist)
                    gate_dist, selector_dist = dist
                    gate = gate_dist.logits.argmax().item()
                    if gate == 0:
                        action = 0
                    else:
                        action = selector_dist.logits.argmax().item() + 1

                obs, _reward, terminated, truncated, info = self.env.step(action)
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

        return tracker
