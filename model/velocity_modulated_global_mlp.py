"""Add zero-initialized time/class modulation to the unchanged Global MLP."""

import torch
from torch import nn

from config.train_config import ModelConfig
from model.velocity_global_mlp import GlobalMLPBlock, VelocityGlobalMLP


class ModulatedGlobalMLPBlock(nn.Module):
    """Reuse baseline branches; modulate their norms with separate scale/shift.

    Condition [B,D] is the sum of projected time and class features. The four
    modulation outputs initially vanish, preserving the baseline function.
    This retains all three main tokens and introduces no Q/K/V attention.
    """

    def __init__(self, base: GlobalMLPBlock, width: int):
        super().__init__()
        # Reuse existing objects rather than reinitialize baseline weights.
        self.mix_norm = base.mix_norm
        self.global_mixer = base.global_mixer
        self.ffn_norm = base.ffn_norm
        self.ffn = base.ffn
        # Constructors draw random weights before zeroing them; restore CPU
        # RNG so the trainer draws exactly the baseline's subsequent samples.
        with torch.random.fork_rng(devices=[]):
            self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 4 * width))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, features: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Refine [B,3,D] features with [B,D] time/class conditioning."""
        scale_mix, shift_mix, scale_ffn, shift_ffn = self.modulation(condition).chunk(4, dim=-1)
        normalized = self.mix_norm(features) * (1 + scale_mix[:, None]) + shift_mix[:, None]
        values, logits = self.global_mixer(normalized.flatten(1)).chunk(2, dim=-1)
        gates = logits.reshape_as(features).softmax(dim=-1) * features.shape[-1]
        hidden = features + values.reshape_as(features) * gates
        normalized = self.ffn_norm(hidden) * (1 + scale_ffn[:, None]) + shift_ffn[:, None]
        return hidden + self.ffn(normalized)


class VelocityModulatedGlobalMLP(VelocityGlobalMLP):
    """Preserve Global MLP initialization and add a direct conditioning branch.

    Construct the complete baseline first, then wrap its existing blocks. This
    keeps base state-dict names, parameter values and initialization RNG state
    identical at the same seed, without loading previously trained weights.
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.blocks = nn.ModuleList([
            ModulatedGlobalMLPBlock(block, config.global_mlp_dim) for block in self.blocks
        ])

    def forward(self, points: torch.Tensor, time: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Predict [B,2] velocity; retain position/time/class in the main state."""
        spatial_phase = (points[:, :, None] * self.spatial_frequencies).flatten(1)
        time_phase = time * self.frequencies
        position = self.position_projection(torch.cat((points, spatial_phase.sin(), spatial_phase.cos()), -1))
        time_features = self.time_projection(torch.cat((time, time_phase.sin(), time_phase.cos()), -1))
        class_features = self.class_projection(self.class_embedding(labels))
        features = torch.stack((position, time_features, class_features), dim=1) + self.token_types
        condition = time_features + class_features
        for block in self.blocks:
            features = block(features, condition)
        return self.output_projection(self.output_norm(features[:, 0]))
