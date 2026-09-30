"""Global MLP with a separate class-conditioning path in every block."""

import math

import torch
from torch import nn

from config.train_config import ModelConfig


class ClassConditionedMLPBlock(nn.Module):
    """Use class-derived affine modulation instead of Q/K/V cross-attention.

    Both pre-normalized residual branches receive scale and shift from the
    class vector. Main features have only two positions: coordinate and time.
    Global Linear mixing and per-position channel softmax gates remain as in
    GlobalMLPBlock. There is no pairwise score matrix or extra class token.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        width = config.global_mlp_dim
        self.class_modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 2 * width))
        self.mix_norm = nn.LayerNorm(width)
        self.global_mixer = nn.Sequential(
            nn.Linear(2 * width, config.global_mlp_bottleneck), nn.SiLU(),
            nn.Linear(config.global_mlp_bottleneck, 4 * width),
        )
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, config.hidden_dim), nn.SiLU(),
                                 nn.Linear(config.hidden_dim, width))

    def forward(self, features: torch.Tensor, class_features: torch.Tensor) -> torch.Tensor:
        """Condition [B,2,D] main features on a separate [B,D] class vector."""
        scale, shift = self.class_modulation(class_features).chunk(2, dim=-1)
        scale, shift = scale[:, None, :], shift[:, None, :]
        # Reuse this block's class modulation in both branches. The +1 keeps
        # the affine map centered on preserving the normalized main features.
        conditioned = self.mix_norm(features) * (1 + scale) + shift
        values, logits = self.global_mixer(conditioned.flatten(1)).chunk(2, dim=-1)
        gates = logits.reshape_as(features).softmax(dim=-1) * features.shape[-1]
        hidden = features + values.reshape_as(features) * gates
        conditioned = self.ffn_norm(hidden) * (1 + scale) + shift
        return hidden + self.ffn(conditioned)


class VelocityClassConditionedMLP(nn.Module):
    """Inject class features at every block without stacking them into the state.

    Class embeddings and projection are shared across blocks; each block has
    its own learned modulation. All computation remains independent per point.
    The dataset, Fourier inputs, final position readout and ODE interface match
    Global MLP. This is FiLM-style conditioning, not standard cross-attention.
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
        self.token_types = nn.Parameter(torch.empty(1, 2, width))
        nn.init.normal_(self.token_types, std=0.02)
        self.blocks = nn.ModuleList([ClassConditionedMLPBlock(config) for _ in range(config.num_layers)])
        self.output_norm = nn.LayerNorm(width)
        self.output_projection = nn.Linear(width, 2)

    def forward(self, points: torch.Tensor, time: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Map [B,2], [B,1], [B] to velocity [B,2] via two main feature vectors."""
        spatial_phase = (points[:, :, None] * self.spatial_frequencies).flatten(1)
        time_phase = time * self.frequencies
        position = self.position_projection(torch.cat((points, spatial_phase.sin(), spatial_phase.cos()), -1))
        time_features = self.time_projection(torch.cat((time, time_phase.sin(), time_phase.cos()), -1))
        class_features = self.class_projection(self.class_embedding(labels))
        features = torch.stack((position, time_features), dim=1) + self.token_types
        for block in self.blocks:
            features = block(features, class_features)
        return self.output_projection(self.output_norm(features[:, 0]))
