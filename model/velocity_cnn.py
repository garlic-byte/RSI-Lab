"""Convolve over position, time and class features to predict pointwise velocity."""

import math

import torch
from torch import nn

from config.train_config import ModelConfig


class VelocityCNN(nn.Module):
    """A Conv1d backbone over three feature positions, not over batch points.

    Uses the same Fourier inputs and class embeddings as the other backbones.
    Each point becomes [channels, 3]; padded width-three convolutions mix its
    position/time/class features. A flattened readout predicts two velocities.
    No batch normalization or dropout couples samples or changes ODE behavior.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        config.validate()
        self.class_embedding = nn.Embedding(config.num_classes, config.embedding_dim)
        self.register_buffer("frequencies", torch.arange(1, config.time_frequencies + 1).float() * math.pi)
        self.register_buffer("spatial_frequencies", torch.arange(1, config.spatial_frequencies + 1).float() * math.pi)
        channels = config.cnn_channels
        self.position_projection = nn.Linear(2 + 4 * config.spatial_frequencies, channels)
        self.time_projection = nn.Linear(1 + 2 * config.time_frequencies, channels)
        self.class_projection = nn.Linear(config.embedding_dim, channels)
        self.feature_types = nn.Parameter(torch.empty(1, channels, 3))
        nn.init.normal_(self.feature_types, std=0.02)
        layers: list[nn.Module] = []
        for _ in range(config.num_layers):
            layers.extend((nn.Conv1d(channels, channels, kernel_size=3, padding=1), nn.SiLU()))
        self.convolutions = nn.Sequential(*layers)
        self.output_projection = nn.Linear(3 * channels, 2)

    def forward(self, points: torch.Tensor, time: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Map [B,2], [B,1], [B] through [B,C,3] features into [B,2]."""
        spatial_phase = (points[:, :, None] * self.spatial_frequencies).flatten(1)
        time_phase = time * self.frequencies
        position = self.position_projection(torch.cat((points, spatial_phase.sin(), spatial_phase.cos()), -1))
        time_features = self.time_projection(torch.cat((time, time_phase.sin(), time_phase.cos()), -1))
        class_features = self.class_projection(self.class_embedding(labels))
        features = torch.stack((position, time_features, class_features), dim=-1) + self.feature_types
        return self.output_projection(self.convolutions(features).flatten(1))
