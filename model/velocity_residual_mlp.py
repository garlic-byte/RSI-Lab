"""Residual-only ablation of the baseline MLP, with identical parameters."""

import torch
from torch import nn

from config.train_config import ModelConfig
from model.velocity_mlp import VelocityMLP


class ResidualMLPNetwork(nn.Sequential):
    """Add identity skips around each hidden-to-hidden Linear + SiLU pair.

    Keep the baseline's ordered modules and state-dict keys. The input
    projection changes width and has no skip; the final velocity readout is
    also unchanged. No normalization, dropout, scaling or new parameters.
    """

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Project features, refine with residual branches, then read velocity."""
        hidden = self[1](self[0](features))
        for index in range(2, len(self) - 1, 2):
            hidden = hidden + self[index + 1](self[index](hidden))
        return self[-1](hidden)


class VelocityResidualMLP(VelocityMLP):
    """Reuse baseline features and initialization; change only skip connections.

    With four hidden layers there is one input projection and three residual
    branches. Reusing modules preserves parameter count and consumes exactly
    the same initialization RNG draws as the baseline MLP.
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.network = ResidualMLPNetwork(*self.network.children())
