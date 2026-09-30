"""Headless plots for target/generated distributions and training history."""

import os
import tempfile
import math
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "torch_shapes_matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from data.shapes import SHAPE_NAMES


def plot_shapes(points: list[torch.Tensor], path: Path, targets: list[torch.Tensor] | None = None,
                generated_label: str = "Generated", shape_names: tuple[str, ...] | list[str] | None = None) -> None:
    """Save a dynamic class gallery, with optional paired target rows."""
    names = SHAPE_NAMES[:len(points)] if shape_names is None else shape_names
    if len(names) != len(points):
        raise ValueError("shape_names must match the number of point clouds")
    columns = 10 if len(points) > 10 else 5
    group_rows = 2 if targets is not None else 1
    rows = math.ceil(len(points) / columns) * group_rows
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 2.8, rows * 2.8), squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)
    for index, name in enumerate(names):
        row, column = divmod(index, columns)
        groups = [(points[index], generated_label)]
        if targets is not None:
            groups = [(targets[index], "Target"), (points[index], "Generated")]
        for offset, (cloud, label) in enumerate(groups):
            ax = axes[row * len(groups) + offset, column]
            ax.set_visible(True)
            array = cloud.detach().cpu().numpy()
            ax.scatter(array[:, 0], array[:, 1], s=1.2, alpha=0.45, linewidths=0)
            ax.set(title=f"{index:02d} {name}\n{label}", xlim=(-1.9, 1.9), ylim=(-1.9, 1.9), aspect="equal")
            ax.set_xticks([])
            ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=120 if len(points) > 10 else 160)
    plt.close(fig)


def plot_history(history: list[dict], path: Path) -> None:
    """Plot logged training/validation losses and learning rate."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    steps = [item["step"] for item in history]
    axes[0].plot(steps, [item["loss"] for item in history], label="Train (window mean)")
    axes[0].plot(steps, [item["validation_loss"] for item in history], label="Fixed validation")
    axes[0].set(xlabel="Step", ylabel="Flow matching MSE")
    axes[0].legend()
    axes[1].plot(steps, [item["learning_rate"] for item in history])
    axes[1].set(xlabel="Step", ylabel="Learning rate")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
