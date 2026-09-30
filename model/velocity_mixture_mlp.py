"""Learned Gaussian-basis flow plus a compact modulated-MLP correction."""

import math

import torch
from torch import nn

from config.train_config import ModelConfig
from model.velocity_modulated_global_mlp import VelocityModulatedGlobalMLP


class VelocityMixtureMLP(VelocityModulatedGlobalMLP):
    """Use a trainable class-conditional density basis inside the velocity model.

    Each label has Gaussian centers, diagonal scales and mixture logits. Their
    Gaussian-to-mixture interpolation has an analytic conditional mean velocity.
    A compact version of the best MLP adds a learned correction. This changes
    only the model: training remains velocity MSE and sampling remains Heun.
    No target samples or shape formulas are used in this module. All classes
    start with the same generic square grid, and all basis parameters are learned.
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        count = config.mixture_components
        side = math.ceil(math.sqrt(count))
        axis = torch.linspace(-1.5, 1.5, side)
        centers = torch.cartesian_prod(axis, axis)
        indices = torch.linspace(0, len(centers) - 1, count).round().long()
        self.component_means = nn.Parameter(centers[indices][None].repeat(config.num_classes, 1, 1))
        self.component_log_scales = nn.Parameter(torch.full((config.num_classes, count, 2), math.log(0.12)))
        self.component_logits = nn.Parameter(torch.zeros(config.num_classes, count))

    def mixture_velocity(self, points: torch.Tensor, time: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Conditional mean of x1-x0 under learned Gaussian mixture endpoints.

        With independent x0~N(0,I), x1|k~N(mu_k,diag(sigma_k^2)), the interpolant
        has mean t*mu_k and variance (1-t)^2 + t^2*sigma_k^2. Responsibilities
        marginalize components; there is no direct sampling from this mixture.
        """
        means = self.component_means[labels]
        target_variance = (2 * self.component_log_scales[labels].clamp(-5, 1)).exp()
        t = time[:, None, :]
        variance = (1 - t).square() + t.square() * target_variance
        delta = points[:, None, :] - t * means
        log_density = self.component_logits[labels].log_softmax(-1)
        log_density = log_density - 0.5 * (delta.square() / variance + variance.log()).sum(-1)
        weights = log_density.softmax(-1)
        conditional = means + (t * target_variance - (1 - t)) / variance * delta
        return (weights[:, :, None] * conditional).sum(1)

    def forward(self, points: torch.Tensor, time: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Return [B,2] velocity; a bounded-time correction vanishes at endpoints."""
        correction = super().forward(points, time, labels)
        return self.mixture_velocity(points, time, labels) + time * (1 - time) * correction
