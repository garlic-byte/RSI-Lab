"""Typed settings shared by training and sampling."""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Architecture settings for models sharing the (points,time,labels) API."""

    model_type: str = "mlp"
    """Registered model name; omitted fields in old checkpoints default to MLP."""
    hidden_dim: int = 128
    num_layers: int = 4
    num_classes: int = 100
    embedding_dim: int = 16
    time_frequencies: int = 8
    spatial_frequencies: int = 6
    transformer_dim: int = 64
    """Transformer token width; hidden_dim sets its feed-forward width."""
    num_heads: int = 4

    def validate(self) -> None:
        """Reject invalid common dimensions before constructing a model."""
        for name in ("hidden_dim", "num_layers", "num_classes", "embedding_dim", "time_frequencies", "spatial_frequencies"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.model_type == "transformer":
            if self.transformer_dim <= 0 or self.num_heads <= 0 or self.transformer_dim % self.num_heads:
                raise ValueError("transformer_dim must be positive and divisible by positive num_heads")


@dataclass
class TrainConfig:
    """Reproducible training and visualization settings."""

    num_classes: int = 100
    """Train the first N catalog classes; 10 selects the checkerboard family."""
    steps: int = 10000
    """Number of optimizer updates; fresh target points are drawn each step."""
    batch_size: int = 1024
    learning_rate: float = 0.001
    min_lr_ratio: float = 0.05
    warmup_steps: int = 200
    weight_decay: float = 0.0001
    grad_clip: float = 1.0
    noise_std: float = 0.035
    """Thickness of the target point clouds in normalized coordinates."""
    seed: int = 42
    device: str = "auto"
    num_threads: int = 4
    output_dir: str = "outputs/default"
    log_every: int = 200
    save_every: int = 1000
    """Interval between progress galleries, independent of log_every."""
    save_progress: bool = False
    """Save an all-class gallery every save_every updates in progress/."""
    gif_duration_ms: int = 500
    """Milliseconds per frame in the automatically generated progress.gif."""
    sample_points: int = 2000
    sample_steps: int = 100
    inference_animation: bool = True
    """Animate final inference for class IDs 9,19,...,99 when present."""
    animation_points: int = 300
    """Maximum fixed particle subset displayed per animated class."""
    inference_gif_duration_ms: int = 50

    def validate(self) -> None:
        """Reject invalid settings before starting an experiment."""
        if self.gif_duration_ms < 10:
            raise ValueError("gif_duration_ms must be at least 10")
        if self.animation_points <= 0 or self.inference_gif_duration_ms < 10:
            raise ValueError("animation_points must be positive and inference_gif_duration_ms >= 10")
        if not 1 <= self.num_classes <= 100:
            raise ValueError("num_classes must be in 1..100")
        for name in ("steps", "batch_size", "num_threads", "log_every", "save_every", "sample_points", "sample_steps"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 <= self.warmup_steps < self.steps:
            raise ValueError("warmup_steps must satisfy 0 <= warmup_steps < steps")
        if self.learning_rate <= 0 or self.grad_clip <= 0 or self.noise_std < 0 or self.weight_decay < 0:
            raise ValueError("Learning rate and grad_clip must be positive; noise/weight decay nonnegative")
        if not 0 < self.min_lr_ratio <= 1:
            raise ValueError("min_lr_ratio must be in (0, 1]")
