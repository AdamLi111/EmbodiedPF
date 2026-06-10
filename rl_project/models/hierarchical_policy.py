"""Hierarchical policy — gate (execute vs friction) then selector (which friction)."""

import torch
import torch.nn as nn
from torch.distributions import Categorical


def _build_mlp(input_dim: int, hidden_dims: list, output_dim: int,
               dropout: float = 0.0) -> nn.Sequential:
    layers = []
    prev = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = h
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)


class HierarchicalPolicy(nn.Module):
    """Two-level policy: gate decides execute-vs-friction, selector picks
    which of the 5 friction types to use."""

    def __init__(self, input_dim: int, config: dict):
        super().__init__()
        pol_cfg = config.get("policy", {})
        hidden_dims = pol_cfg.get("hidden_dims", [256, 128])
        dropout = pol_cfg.get("dropout", 0.0)

        self.gate_network = _build_mlp(input_dim, hidden_dims, 2, dropout)
        self.selector_network = _build_mlp(input_dim, hidden_dims, 5, dropout)

    def forward(self, state: torch.Tensor):
        gate_dist = Categorical(logits=self.gate_network(state))
        selector_dist = Categorical(logits=self.selector_network(state))
        return gate_dist, selector_dist

    def get_action(self, state: torch.Tensor):
        gate_dist, selector_dist = self.forward(state)
        gate = gate_dist.sample()
        gate_lp = gate_dist.log_prob(gate)

        if gate.item() == 0:
            return 0, gate_lp.item()

        selector = selector_dist.sample()
        selector_lp = selector_dist.log_prob(selector)
        action = selector.item() + 1  # actions 1-5
        return action, (gate_lp + selector_lp).item()

    def evaluate_action(self, state: torch.Tensor, action: torch.Tensor):
        gate_dist, selector_dist = self.forward(state)

        # gate target: 0 if action==0, else 1
        gate_target = (action > 0).long()
        gate_lp = gate_dist.log_prob(gate_target)
        gate_ent = gate_dist.entropy()

        # For execute actions (action==0) selector doesn't contribute
        is_friction = (action > 0).float()
        selector_target = torch.clamp(action - 1, min=0)
        selector_lp = selector_dist.log_prob(selector_target)
        selector_ent = selector_dist.entropy()

        log_prob = gate_lp + is_friction * selector_lp
        entropy = gate_ent + is_friction * selector_ent
        return log_prob, entropy
