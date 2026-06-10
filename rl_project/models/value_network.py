"""Value network — maps encoded state to a scalar value estimate."""

import torch
import torch.nn as nn


class ValueNetwork(nn.Module):
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
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)
