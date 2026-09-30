"""Reload a trained checkpoint and generate fresh point-cloud images."""

import argparse
from pathlib import Path

import torch

from config.train_config import ModelConfig
from data.shapes import LEGACY_VERSION, get_shape_names
from model.registry import build_model
from train.sampling import generate_points
from utils.plotting import plot_shapes
from utils.device import resolve_device
from utils.inference_animation import trajectory_recorder, save_inference_animation


def main() -> None:
    """Sample all checkpoint classes using the checkpoint architecture."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/default/checkpoint.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/samples"))
    parser.add_argument("--points", type=int, default=2000)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--inference-animation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--animation-points", type=int, default=300)
    parser.add_argument("--inference-gif-duration-ms", type=int, default=50)
    args = parser.parse_args()
    if args.points <= 0 or args.steps <= 0:
        parser.error("points and steps must be positive")
    if args.animation_points <= 0 or args.inference_gif_duration_ms < 10:
        parser.error("animation-points must be positive and inference-gif-duration-ms >= 10")
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model_config = ModelConfig(**checkpoint["config"]["model"])
    model = build_model(model_config).to(device)
    model.load_state_dict(checkpoint["model"])
    dataset_version = checkpoint["config"].get("dataset_version", LEGACY_VERSION)
    shape_names = checkpoint["config"].get("shape_names", get_shape_names(dataset_version)[:model_config.num_classes])
    clouds = []
    trajectories = {}
    for class_id in range(model_config.num_classes):
        labels = torch.full((args.points,), class_id, dtype=torch.long, device=device)
        recorder = (trajectory_recorder(trajectories, class_id, args.animation_points)
                    if args.inference_animation and class_id % 10 == 9 else None)
        clouds.append(generate_points(model, labels, args.steps, on_step=recorder).cpu())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_shapes(clouds, args.output_dir / "generated.png", shape_names=shape_names)
    torch.save({"generated": torch.stack(clouds), "shape_names": shape_names,
                "dataset_version": dataset_version}, args.output_dir / "points.pt")
    save_inference_animation(trajectories, shape_names, args.output_dir, args.inference_gif_duration_ms, dataset_version)
    print(f"Saved samples to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
