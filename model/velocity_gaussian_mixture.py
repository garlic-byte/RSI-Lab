"""Density-consistent Gaussian mixture velocity without an MLP correction."""

import math

import torch
from torch import nn

from config.train_config import ModelConfig
from model.velocity_correlated_mixture_mlp import VelocityCorrelatedMixtureMLP


class VelocityGaussianMixture(VelocityCorrelatedMixtureMLP):
    """Allocate the full model budget to learned full-covariance density bases.

    Removing the free MLP correction makes all times describe the same learned
    endpoint density. Components are still trained only through velocity MSE;
    inference still integrates their velocity with the existing Heun sampler.
    """

    def __init__(self, config: ModelConfig):
        # Deliberately omit the parent MLP: no unused parameters count toward
        # the budget. Every parameter below participates in the velocity field.
        nn.Module.__init__(self)
        count = config.mixture_components
        side = math.ceil(math.sqrt(count))
        axis = torch.linspace(-1.5, 1.5, side)
        centers = torch.cartesian_prod(axis, axis)
        indices = torch.linspace(0, len(centers) - 1, count).round().long()
        self.component_means = nn.Parameter(centers[indices][None].repeat(config.num_classes, 1, 1))
        self.component_log_scales = nn.Parameter(torch.full((config.num_classes, count, 2), math.log(0.12)))
        self.component_logits = nn.Parameter(torch.zeros(config.num_classes, count))
        self.component_correlations = nn.Parameter(torch.zeros(config.num_classes, count))

    def forward(self, points: torch.Tensor, time: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Return the analytic posterior-mean velocity of the learned mixture."""
        return self.mixture_velocity(points, time, labels)
