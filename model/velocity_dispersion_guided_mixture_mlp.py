"""Expose between-component velocity dispersion to the correction MLP."""

import torch
from torch import nn

from config.train_config import ModelConfig
from model.velocity_guided_mixture_mlp import VelocityGuidedMixtureMLP


class VelocityDispersionGuidedMixtureMLP(VelocityGuidedMixtureMLP):
    """Supplement the mean velocity with three local second-moment features.

    These are moments of component conditional means, not a claim about total
    posterior uncertainty. They describe whether mixture components agree on
    the local direction. No samples beyond the current input are inspected.
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        with torch.random.fork_rng(devices=[]):
            self.dispersion_projection = nn.Linear(3, config.global_mlp_dim, bias=False)
        nn.init.zeros_(self.dispersion_projection.weight)

    def compute_guidance(self, points: torch.Tensor, time: torch.Tensor,
                         labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the mean and centered weighted covariance in one basis pass."""
        conditional, weights = self.component_velocity_fields(points, time, labels)
        velocity = (weights[..., None] * conditional).sum(1)
        centered = conditional - velocity[:, None, :]
        variance = (weights[..., None] * centered.square()).sum(1)
        cross = (weights * centered.prod(-1)).sum(1)
        normalized_cross = cross / (1 + variance).prod(-1).sqrt()
        features = torch.cat((variance.log1p(), normalized_cross[:, None]), -1)
        guidance = self.velocity_projection(velocity) + self.dispersion_projection(features)
        return velocity, guidance
