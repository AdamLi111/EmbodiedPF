"""Simple rollout buffer with GAE advantage computation."""

import torch
import numpy as np


class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.advantages = None
        self.returns = None

    def add(self, state, action, reward, done, log_prob, value):
        self.states.append(state.detach() if isinstance(state, torch.Tensor)
                           else torch.as_tensor(state, dtype=torch.float32))
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def compute_returns_and_advantages(self, last_value: float, gamma: float,
                                       gae_lambda: float):
        n = len(self.rewards)
        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0.0
        values = [v if isinstance(v, float) else v.item()
                  for v in self.values] + [last_value]

        for t in reversed(range(n)):
            next_non_terminal = 1.0 - float(self.dones[t])
            delta = (self.rewards[t]
                     + gamma * values[t + 1] * next_non_terminal
                     - values[t])
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        self.advantages = torch.tensor(advantages, dtype=torch.float32)
        self.returns = self.advantages + torch.tensor(
            values[:-1], dtype=torch.float32
        )

    def get_batches(self, num_minibatches: int):
        n = len(self.states)
        indices = np.arange(n)
        np.random.shuffle(indices)
        batch_size = max(n // num_minibatches, 1)

        states = torch.stack(self.states)
        actions = torch.tensor(self.actions, dtype=torch.long)
        old_log_probs = torch.tensor(self.log_probs, dtype=torch.float32)

        for start in range(0, n, batch_size):
            idx = indices[start:start + batch_size]
            yield (
                states[idx],
                actions[idx],
                old_log_probs[idx],
                self.returns[idx],
                self.advantages[idx],
            )

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()
        self.advantages = None
        self.returns = None
