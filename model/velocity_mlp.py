"""Predict a two-dimensional flow velocity from position, time and class."""

import math

import torch
from torch import nn

from config.train_config import ModelConfig


class VelocityMLP(nn.Module):
    """An MLP backbone with class embeddings and Fourier time features."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.class_embedding = nn.Embedding(config.num_classes, config.embedding_dim)
        self.register_buffer("frequencies", torch.arange(1, config.time_frequencies + 1).float() * math.pi)
        self.register_buffer("spatial_frequencies", torch.arange(1, config.spatial_frequencies + 1).float() * math.pi)
        input_dim = 2 + 4 * config.spatial_frequencies + 1 + 2 * config.time_frequencies + config.embedding_dim
        layers: list[nn.Module] = []
        for _ in range(config.num_layers):
            layers.extend((nn.Linear(input_dim, config.hidden_dim), nn.SiLU()))
            input_dim = config.hidden_dim
        layers.append(nn.Linear(input_dim, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, points: torch.Tensor, time: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Map points [B,2], time [B,1], labels [B] to velocity [B,2]."""
        phase = time * self.frequencies
        # Fixed position features let a small MLP resolve separated narrow bands.
        spatial_phase = (points[:, :, None] * self.spatial_frequencies).flatten(1)
        features = torch.cat((points, spatial_phase.sin(), spatial_phase.cos(), time, phase.sin(), phase.cos(), self.class_embedding(labels)), dim=-1)
        return self.network(features)
