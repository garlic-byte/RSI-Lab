"""Integrate learned velocities to transform Gaussian noise into point clouds."""

import torch
from collections.abc import Callable

from torch import nn


@torch.inference_mode()
def generate_points(model: nn.Module, labels: torch.Tensor, steps: int = 100,
                    initial_noise: torch.Tensor | None = None,
                    on_step: Callable[[int, torch.Tensor], None] | None = None) -> torch.Tensor:
    """Integrate with Heun on [0,1]; optionally reuse fixed Gaussian noise.

    initial_noise is cloned so repeated previews never modify their starting
    points or consume training random numbers.
    """
    if steps <= 0:
        raise ValueError("Sampling steps must be positive")
    if initial_noise is not None and (initial_noise.shape != (len(labels), 2) or initial_noise.device != labels.device):
        raise ValueError("initial_noise must have shape [len(labels), 2] on the labels device")
    was_training = model.training
    model.eval()
    points = torch.randn(labels.shape[0], 2, device=labels.device) if initial_noise is None else initial_noise.clone()
    dt = 1.0 / steps
    if on_step is not None:
        on_step(0, points)
    for step in range(steps):
        time = torch.full((len(labels), 1), step * dt, device=labels.device)
        velocity = model(points, time, labels)
        next_velocity = model(points + dt * velocity, time + dt, labels)
        points = points + 0.5 * dt * (velocity + next_velocity)
        if on_step is not None:
            on_step(step + 1, points)
    model.train(was_training)
    return points
