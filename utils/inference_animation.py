"""Animate the identities of sampled particles through actual Heun steps."""

import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from utils.plotting import plt


def trajectory_recorder(storage: dict[int, list[torch.Tensor]], class_id: int, point_limit: int):
    """Capture the same leading particles at every step, without sampling RNG."""
    storage[class_id] = []

    def record(step: int, points: torch.Tensor) -> None:
        if step != len(storage[class_id]):
            raise ValueError("Trajectory steps must be contiguous, starting at zero")
        storage[class_id].append(points[:point_limit].detach().cpu().clone())

    return record


def save_inference_animation(trajectories: dict[int, list[torch.Tensor]], shape_names: list[str] | tuple[str, ...],
                             output_dir: Path, duration_ms: int = 50, dataset_version: str = "unknown") -> None:
    """Render initial noise plus every accepted solver step, retaining point IDs.

    Positions are actual solver states, not interpolated images. Point colors
    encode initial angle and remain fixed. Axis limits include the entire path.
    """
    if not trajectories:
        return
    if duration_ms < 10:
        raise ValueError("inference GIF duration must be at least 10ms")
    class_ids = sorted(trajectories)
    paths = torch.stack([torch.stack(trajectories[index]) for index in class_ids])
    if not torch.isfinite(paths).all():
        raise ValueError("Cannot animate non-finite particle coordinates")
    frame_count = paths.shape[1]
    steps = frame_count - 1
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"positions": paths, "class_ids": class_ids,
                "shape_names": [shape_names[i] for i in class_ids],
                "times": torch.linspace(0, 1, frame_count), "solver": "Heun",
                "dataset_version": dataset_version}, output_dir / "inference_trajectories.pt")
    columns = min(5, len(class_ids))
    rows = math.ceil(len(class_ids) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 3, rows * 3.1), squeeze=False)
    limit = max(1.9, math.ceil(paths.abs().max().item() * 1.05 * 2) / 2)
    scatters = []
    for ax in axes.flat:
        ax.set_visible(False)
    for index, class_id in enumerate(class_ids):
        ax = axes.flat[index]
        ax.set_visible(True)
        initial = paths[index, 0].numpy()
        colors = (np.arctan2(initial[:, 1], initial[:, 0]) + np.pi) / (2 * np.pi)
        scatters.append(ax.scatter(initial[:, 0], initial[:, 1], c=colors, cmap="hsv", vmin=0, vmax=1,
                                   s=4, alpha=.7, linewidths=0))
        ax.set(title=f"{class_id:02d} {shape_names[class_id]}", xlim=(-limit, limit), ylim=(-limit, limit), aspect="equal")
        ax.set_xticks([])
        ax.set_yticks([])
    title = fig.suptitle("")
    fig.subplots_adjust(left=.025, right=.985, bottom=.035, top=.86, wspace=.15, hspace=.3)
    frames = []
    try:
        for step in range(frame_count):
            for index, scatter in enumerate(scatters):
                scatter.set_offsets(paths[index, step].numpy())
            title.set_text(f"Gaussian noise -> generated points | Heun step {step}/{steps} | t={step / max(steps, 1):.2f}")
            fig.canvas.draw()
            frame = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert("RGB")
            if step in (0, steps):
                frame.save(output_dir / ("inference_start.png" if step == 0 else "inference_end.png"))
            frames.append(frame.convert("P", palette=Image.Palette.ADAPTIVE))
            frame.close()
        # A short hold makes the initial/final distributions easier to compare.
        durations = [duration_ms] * frame_count
        durations[0] = max(500, duration_ms)
        durations[-1] = max(1000, duration_ms)
        frames[0].save(output_dir / "inference.gif", save_all=True, append_images=frames[1:],
                       duration=durations, loop=0, disposal=2, optimize=False)
    finally:
        plt.close(fig)
        for frame in frames:
            frame.close()
    print(f"Saved inference animation: {output_dir / 'inference.gif'}", flush=True)
