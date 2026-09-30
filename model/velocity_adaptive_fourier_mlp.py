"""Class-adaptive two-dimensional Fourier inputs for the modulated Global MLP."""

import math

import torch
from torch import nn

from config.train_config import ModelConfig
from model.velocity_modulated_global_mlp import VelocityModulatedGlobalMLP


class VelocityAdaptiveFourierMLP(VelocityModulatedGlobalMLP):
    """Learn plane-wave directions/frequencies per label, keeping the backbone.

    Every class starts with the same four directions and increasing harmonics;
    initialization uses no shape formulas or target samples. The phase count
    is identical to the baseline's separate x/y Fourier encoding. Only the
    spatial feature basis changes: [B,2] times [B,2F,2] gives [B,2F].
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        modes = 2 * config.spatial_frequencies
        indices = torch.arange(modes)
        angles = (indices % 4).float() * (math.pi / 4)
        harmonics = (indices // 4 + 1).float() * math.pi
        basis = torch.stack((angles.cos(), angles.sin()), dim=-1) * harmonics[:, None]
        self.spatial_directions = nn.Parameter(basis[None].repeat(config.num_classes, 1, 1))

    def forward(self, points: torch.Tensor, time: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Predict [B,2] velocity with class-specific learned spatial features."""
        directions = self.spatial_directions[labels]
        spatial_phase = (points[:, None, :] * directions).sum(dim=-1)
        time_phase = time * self.frequencies
        position = self.position_projection(torch.cat((points, spatial_phase.sin(), spatial_phase.cos()), -1))
        time_features = self.time_projection(torch.cat((time, time_phase.sin(), time_phase.cos()), -1))
        class_features = self.class_projection(self.class_embedding(labels))
        features = torch.stack((position, time_features, class_features), dim=1) + self.token_types
        condition = time_features + class_features
        for block in self.blocks:
            features = block(features, condition)
        return self.output_projection(self.output_norm(features[:, 0]))
