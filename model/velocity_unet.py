"""A small one-dimensional U-Net for conditional pointwise flow matching."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from config.train_config import ModelConfig


def convolution_block(in_channels: int, out_channels: int) -> nn.Sequential:
    """Two padded convolutions preserve sequence length; no batch coupling."""
    return nn.Sequential(nn.Conv1d(in_channels, out_channels, 3, padding=1), nn.SiLU(),
                         nn.Conv1d(out_channels, out_channels, 3, padding=1), nn.SiLU())


class VelocityUNet(nn.Module):
    """Encode lengths 3 -> 2 -> 1, then decode 1 -> 2 -> 3 with skip features.

    The sequence contains position/time/class features for ONE point, not
    neighboring cloud points or image pixels. Channels increase C -> 2C -> 4C.
    Decoder concatenation retains encoder detail at each matching resolution.
    No normalization or dropout is added in this architecture experiment.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        config.validate()
        channels = config.unet_channels
        self.class_embedding = nn.Embedding(config.num_classes, config.embedding_dim)
        self.register_buffer("frequencies", torch.arange(1, config.time_frequencies + 1).float() * math.pi)
        self.register_buffer("spatial_frequencies", torch.arange(1, config.spatial_frequencies + 1).float() * math.pi)
        self.position_projection = nn.Linear(2 + 4 * config.spatial_frequencies, channels)
        self.time_projection = nn.Linear(1 + 2 * config.time_frequencies, channels)
        self.class_projection = nn.Linear(config.embedding_dim, channels)
        self.feature_types = nn.Parameter(torch.empty(1, channels, 3))
        nn.init.normal_(self.feature_types, std=0.02)
        self.encoder_0 = convolution_block(channels, channels)
        self.down_0 = nn.Conv1d(channels, 2 * channels, 3, stride=2, padding=1)
        self.encoder_1 = convolution_block(2 * channels, 2 * channels)
        self.down_1 = nn.Conv1d(2 * channels, 4 * channels, 3, stride=2, padding=1)
        self.bottleneck = convolution_block(4 * channels, 4 * channels)
        self.up_1 = nn.Conv1d(4 * channels, 2 * channels, 1)
        self.decoder_1 = convolution_block(4 * channels, 2 * channels)
        self.up_0 = nn.Conv1d(2 * channels, channels, 1)
        self.decoder_0 = convolution_block(2 * channels, channels)
        self.output_projection = nn.Linear(3 * channels, 2)

    def forward(self, points: torch.Tensor, time: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Map [B,2], [B,1], [B] to [B,2] with per-point encoder/decoder skips."""
        spatial_phase = (points[:, :, None] * self.spatial_frequencies).flatten(1)
        time_phase = time * self.frequencies
        position = self.position_projection(torch.cat((points, spatial_phase.sin(), spatial_phase.cos()), -1))
        time_features = self.time_projection(torch.cat((time, time_phase.sin(), time_phase.cos()), -1))
        class_features = self.class_projection(self.class_embedding(labels))
        features = torch.stack((position, time_features, class_features), dim=-1) + self.feature_types
        skip_0 = self.encoder_0(features)
        skip_1 = self.encoder_1(F.silu(self.down_0(skip_0)))
        hidden = self.bottleneck(F.silu(self.down_1(skip_1)))
        # Match saved lengths explicitly: doubling 2 would overshoot length 3.
        hidden = self.up_1(F.interpolate(hidden, size=skip_1.shape[-1], mode="nearest"))
        hidden = self.decoder_1(torch.cat((hidden, skip_1), dim=1))
        hidden = self.up_0(F.interpolate(hidden, size=skip_0.shape[-1], mode="nearest"))
        hidden = self.decoder_0(torch.cat((hidden, skip_0), dim=1))
        return self.output_projection(hidden.flatten(1))
