"""Global dense feature mixing with normalization, residuals and channel gates."""

import math

import torch
from torch import nn

from config.train_config import ModelConfig


class GlobalMLPBlock(nn.Module):
    """Mix all three feature vectors with dense layers, without pairwise attention.

    Two pre-normalized residual branches mirror the Transformer block layout.
    The first flattens all tokens and computes features and channel gates from
    the entire input. Softmax is over channels within each token, NOT keys.
    Gates multiply features elementwise; there is no QK product or AV mixing.
    The second is the same shared two-layer SiLU FFN used by the Transformer.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        width = config.global_mlp_dim
        self.mix_norm = nn.LayerNorm(width)
        self.global_mixer = nn.Sequential(
            nn.Linear(3 * width, config.global_mlp_bottleneck), nn.SiLU(),
            nn.Linear(config.global_mlp_bottleneck, 6 * width),
        )
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, config.hidden_dim), nn.SiLU(),
                                 nn.Linear(config.hidden_dim, width))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Refine [B,3,D] features with global dense mixing and channel gating."""
        mixed = self.global_mixer(self.mix_norm(features).flatten(1))
        values, logits = mixed.chunk(2, dim=-1)
        values = values.reshape_as(features)
        logits = logits.reshape_as(features)
        # D * softmax gives mean gate 1, instead of shrinking the branch by D.
        # This is channel gating, not the 1/sqrt(head_dim) attention scaling.
        gates = logits.softmax(dim=-1) * features.shape[-1]
        hidden = features + values * gates
        return hidden + self.ffn(self.ffn_norm(hidden))


class VelocityGlobalMLP(nn.Module):
    """Match Transformer input/readout structure with an attention-free backbone.

    Same Fourier features, class embeddings and three learned feature types.
    Every global mixer sees all 3*D features of a point; samples stay independent.
    This combined-operator experiment is not a pure normalization ablation.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        config.validate()
        width = config.global_mlp_dim
        self.class_embedding = nn.Embedding(config.num_classes, config.embedding_dim)
        self.register_buffer("frequencies", torch.arange(1, config.time_frequencies + 1).float() * math.pi)
        self.register_buffer("spatial_frequencies", torch.arange(1, config.spatial_frequencies + 1).float() * math.pi)
        self.position_projection = nn.Linear(2 + 4 * config.spatial_frequencies, width)
        self.time_projection = nn.Linear(1 + 2 * config.time_frequencies, width)
        self.class_projection = nn.Linear(config.embedding_dim, width)
        self.token_types = nn.Parameter(torch.empty(1, 3, width))
        nn.init.normal_(self.token_types, std=0.02)
        self.blocks = nn.ModuleList([GlobalMLPBlock(config) for _ in range(config.num_layers)])
        self.output_norm = nn.LayerNorm(width)
        self.output_projection = nn.Linear(width, 2)

    def forward(self, points: torch.Tensor, time: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Map [B,2], [B,1], [B] into [B,2] using global Linear feature mixing."""
        spatial_phase = (points[:, :, None] * self.spatial_frequencies).flatten(1)
        time_phase = time * self.frequencies
        position = self.position_projection(torch.cat((points, spatial_phase.sin(), spatial_phase.cos()), -1))
        time_features = self.time_projection(torch.cat((time, time_phase.sin(), time_phase.cos()), -1))
        class_features = self.class_projection(self.class_embedding(labels))
        features = torch.stack((position, time_features, class_features), dim=1) + self.token_types
        for block in self.blocks:
            features = block(features)
        return self.output_projection(self.output_norm(features[:, 0]))
