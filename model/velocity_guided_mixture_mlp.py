"""Condition the hybrid correction on its learned Gaussian velocity branch."""

import torch
from torch import nn

from config.train_config import ModelConfig
from model.velocity_oriented_mixture_mlp import VelocityOrientedMixtureMLP


class VelocityGuidedMixtureMLP(VelocityOrientedMixtureMLP):
    """Expose local basis geometry to the MLP instead of only adding outputs.

    A zero-initialized projection preserves the original initial function and
    RNG stream. The guidance is computed from current learned parameters and
    the same input point, never from target samples or neighboring particles.
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        with torch.random.fork_rng(devices=[]):
            self.velocity_projection = nn.Linear(2, config.global_mlp_dim, bias=False)
        nn.init.zeros_(self.velocity_projection.weight)

    def guide_condition(self, condition: torch.Tensor, basis_velocity: torch.Tensor) -> torch.Tensor:
        """Keep the baseline time/class condition; variants may add geometry."""
        return condition

    def forward(self, points: torch.Tensor, time: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Return velocity with a geometry-conditioned residual correction."""
        basis_velocity = self.mixture_velocity(points, time, labels)
        spatial_phase = (points[:, :, None] * self.spatial_frequencies).flatten(1)
        time_phase = time * self.frequencies
        position = self.position_projection(torch.cat((points, spatial_phase.sin(), spatial_phase.cos()), -1))
        position = position + self.velocity_projection(basis_velocity)
        time_features = self.time_projection(torch.cat((time, time_phase.sin(), time_phase.cos()), -1))
        class_features = self.class_projection(self.class_embedding(labels))
        features = torch.stack((position, time_features, class_features), dim=1) + self.token_types
        condition = self.guide_condition(time_features + class_features, basis_velocity)
        for block in self.blocks:
            features = block(features, condition)
        correction = self.output_projection(self.output_norm(features[:, 0]))
        return basis_velocity + time * (1 - time) * correction
