"""Inject Gaussian velocity into every residual block's modulation condition."""

import torch
from torch import nn

from config.train_config import ModelConfig
from model.velocity_guided_mixture_mlp import VelocityGuidedMixtureMLP


class VelocityBlockGuidedMixtureMLP(VelocityGuidedMixtureMLP):
    """Extend input guidance with geometry-conditioned scales and shifts.

    The new projection starts at zero and preserves initialization RNG, so the
    experiment starts from the exact same function as the input-guided model.
    The unchanged modulation modules share the resulting condition across blocks.
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        with torch.random.fork_rng(devices=[]):
            self.velocity_condition_projection = nn.Linear(2, config.global_mlp_dim, bias=False)
        nn.init.zeros_(self.velocity_condition_projection.weight)

    def guide_condition(self, condition: torch.Tensor, basis_velocity: torch.Tensor) -> torch.Tensor:
        """Add learned local-velocity features to the time/class condition."""
        return condition + self.velocity_condition_projection(basis_velocity)
