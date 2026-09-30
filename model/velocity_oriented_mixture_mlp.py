"""Principal-axis Gaussian components with a compact modulated MLP."""

import math

import torch
from torch import nn

from config.train_config import ModelConfig
from model.velocity_correlated_mixture_mlp import VelocityCorrelatedMixtureMLP


class VelocityOrientedMixtureMLP(VelocityCorrelatedMixtureMLP):
    """Learn ellipse angles and principal widths directly for thin local supports.

    The model retains the best hybrid's MLP and component count. Each class
    receives the same generic grid and alternating orientations. Unequal initial
    widths give the angle a gradient from the first optimization step.
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        del self.component_correlations
        angles = (torch.arange(config.mixture_components) % 8).float() * (math.pi / 8)
        self.component_angles = nn.Parameter(angles[None].repeat(config.num_classes, 1))
        with torch.no_grad():
            self.component_log_scales[..., 1].fill_(math.log(0.06))

    def target_covariance(self, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Rotate positive principal variances into the original xy coordinates."""
        eigenvalues = (2 * self.component_log_scales[labels].clamp(-5, 1)).exp()
        angles = self.component_angles[labels]
        cosine, sine = angles.cos(), angles.sin()
        major, minor = eigenvalues.unbind(-1)
        diagonal = torch.stack((major * cosine.square() + minor * sine.square(),
                                major * sine.square() + minor * cosine.square()), -1)
        covariance = (major - minor) * cosine * sine
        return diagonal, covariance
