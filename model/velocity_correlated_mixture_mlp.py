"""Full-covariance Gaussian velocity bases with a modulated MLP correction."""

import torch
from torch import nn

from config.train_config import ModelConfig
from model.velocity_mixture_mlp import VelocityMixtureMLP


class VelocityCorrelatedMixtureMLP(VelocityMixtureMLP):
    """Allow learned Gaussian components to align with curved local supports.

    A bounded correlation extends each diagonal basis to a positive-definite
    2D covariance. Initialization remains identical across classes and uses
    no target geometry. The inherited forward retains the same MLP correction.
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.component_correlations = nn.Parameter(torch.zeros(config.num_classes, config.mixture_components))

    def target_covariance(self, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return diagonal and off-diagonal entries of each endpoint covariance."""
        scales = self.component_log_scales[labels].clamp(-5, 1).exp()
        target_variance = scales.square()
        covariance = 0.99 * self.component_correlations[labels].tanh() * scales.prod(-1)
        return target_variance, covariance

    def mixture_velocity(self, points: torch.Tensor, time: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Return posterior-averaged velocity using explicit 2x2 covariance solves."""
        conditional, weights = self.component_velocity_fields(points, time, labels)
        return (weights[..., None] * conditional).sum(1)

    def component_velocity_fields(self, points: torch.Tensor, time: torch.Tensor,
                                  labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return component conditional means [B,K,2] and responsibilities [B,K]."""
        means = self.component_means[labels]
        target_variance, covariance = self.target_covariance(labels)
        t = time[:, None, :]
        variance = (1 - t).square() + t.square() * target_variance
        cross = t.squeeze(-1).square() * covariance
        determinant = variance.prod(-1) - cross.square()
        delta = points[:, None, :] - t * means
        # Explicit inverse avoids a batched matrix solver in every MPS call.
        solved = torch.stack((variance[..., 1] * delta[..., 0] - cross * delta[..., 1],
                              variance[..., 0] * delta[..., 1] - cross * delta[..., 0]), -1)
        solved = solved / determinant[..., None]
        log_density = self.component_logits[labels].log_softmax(-1)
        log_density = log_density - 0.5 * ((delta * solved).sum(-1) + determinant.log())
        weights = log_density.softmax(-1)
        coefficient = t * target_variance - (1 - t)
        cross_coefficient = t * covariance[..., None]
        conditional = means + coefficient * solved + cross_coefficient * solved.flip(-1)
        return conditional, weights
