"""Flat MLP policy — maps encoded state to a 6-way Categorical over actions."""

import torch
import torch.nn as nn
from torch.distributions import Categorical


class FlatPolicy(nn.Module):
    """Simple feedforward policy: state → hidden layers → 6-action softmax."""

    def __init__(self, input_dim: int, config: dict):
        super().__init__()
        pol_cfg = config.get("policy", {})
        hidden_dims = pol_cfg.get("hidden_dims", [256, 128])
        dropout = pol_cfg.get("dropout", 0.0)

        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 6))
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> Categorical:
        logits = self.net(state)
        return Categorical(logits=logits)

    def get_action(self, state: torch.Tensor):
        dist = self.forward(state)
        action = dist.sample()
        return action.item(), dist.log_prob(action).item()

    def evaluate_action(self, state: torch.Tensor, action: torch.Tensor):
        dist = self.forward(state)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, entropy
