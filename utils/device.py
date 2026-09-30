"""Resolve explicit devices or prefer available GPU backends."""

import torch


def resolve_device(requested: str) -> torch.device:
    """Prefer CUDA, then Apple MPS, with CPU as the automatic fallback."""
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable in this process. Check GPU permissions or run from macOS Terminal.")
    return torch.device(requested)


def synchronize_device(device: torch.device) -> None:
    """Wait for GPU work before reporting wall-clock training duration."""
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)
