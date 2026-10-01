"""Augment local velocity moments with learned density and responsibility entropy."""

import math

import torch
from torch import nn

from config.train_config import ModelConfig
from model.velocity_dispersion_guided_mixture_mlp import VelocityDispersionGuidedMixtureMLP


class VelocityDensityGuidedMixtureMLP(VelocityDispersionGuidedMixtureMLP):
    """Expose local density magnitude and component overlap to the correction.

    Features come entirely from the current model's Gaussian interpolation.
    They are not target-density labels and add no supervision or external input.
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        with torch.random.fork_rng(devices=[]):
            self.density_projection = nn.Linear(2, config.global_mlp_dim, bias=False)
        nn.init.zeros_(self.density_projection.weight)

    def compute_guidance(self, points: torch.Tensor, time: torch.Tensor,
                         labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute all guidance from one component evaluation."""
        conditional, weights, log_components = self.component_velocity_statistics(points, time, labels)
        velocity, guidance = self.project_velocity_moments(conditional, weights)
        log_density = log_components.logsumexp(-1) - math.log(2 * math.pi)
        entropy = -(weights * weights.clamp_min(1e-12).log()).sum(-1)
        entropy = entropy / math.log(max(weights.shape[-1], 2))
        features = torch.stack((log_density.asinh(), entropy), -1)
        return velocity, guidance + self.density_projection(features)
