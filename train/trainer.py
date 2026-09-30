"""Conditional flow matching with AdamW, warmup and cosine decay."""

import csv
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from config.train_config import ModelConfig, TrainConfig
from data.shapes import DATASET_VERSION, SHAPE_NAMES, sample_batch, sample_shape
from model.registry import build_model
from train.sampling import generate_points
from utils.plotting import plot_history, plot_shapes
from utils.device import resolve_device, synchronize_device
from utils.animation import build_progress_gif
from evaluation.quality import evaluate_clouds
from utils.inference_animation import trajectory_recorder, save_inference_animation


def learning_rate_factor(step: int, config: TrainConfig) -> float:
    """Scale the next update's LR: linear warmup followed by cosine decay."""
    if step < config.warmup_steps:
        return (step + 1) / max(1, config.warmup_steps)
    progress = (step - config.warmup_steps) / max(1, config.steps - config.warmup_steps - 1)
    progress = min(progress, 1.0)
    return config.min_lr_ratio + (1 - config.min_lr_ratio) * (1 + math.cos(math.pi * progress)) / 2


def flow_batch(count: int, config: TrainConfig, device: torch.device) -> tuple[torch.Tensor, ...]:
    """Pair independent Gaussian and target draws along a straight path."""
    # CPU shape construction avoids ten dynamic mask/synchronization operations
    # on MPS per update; dense model forward/backward still runs on the GPU.
    target, labels = sample_batch(count, config.noise_std, "cpu", config.num_classes)
    target, labels = target.to(device), labels.to(device)
    source = torch.randn_like(target)
    times = torch.rand(count, 1, device=device)
    return torch.lerp(source, target, times), times, labels, target - source


@torch.inference_mode()
def sliced_wasserstein(generated: torch.Tensor, target: torch.Tensor) -> float:
    """Compare equally sized clouds using 128 deterministic 1D projections."""
    angles = torch.linspace(0, math.pi, 129, device=generated.device)[:-1]
    directions = torch.stack((angles.cos(), angles.sin()))
    a = (generated @ directions).sort(dim=0).values
    b = (target @ directions).sort(dim=0).values
    return (a - b).abs().mean().item()


def run_training(config: TrainConfig, model_config: ModelConfig) -> None:
    """Train, save reloadable weights, then render all configured generated classes."""
    config.validate()
    if model_config.num_classes != config.num_classes:
        raise ValueError("Model and training num_classes must match")
    shape_names = SHAPE_NAMES[:config.num_classes]
    torch.set_num_threads(config.num_threads)
    torch.manual_seed(config.seed)
    device = resolve_device(config.device)
    output_dir = Path(config.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = {"train": asdict(config), "model": asdict(model_config), "shape_names": shape_names,
                "dataset_version": DATASET_VERSION, "resolved_device": str(device)}
    (output_dir / "config.json").write_text(json.dumps(settings, indent=2) + "\n")

    model = build_model(model_config).to(device)
    progress_noise = None
    progress_paths: list[Path] = []
    if config.save_progress:
        (output_dir / "progress").mkdir(parents=True, exist_ok=True)
        # A private CPU generator keeps both training RNG and cross-step noise
        # unchanged, so visual differences reflect the learned velocity field.
        generator = torch.Generator(device="cpu").manual_seed(config.seed + 2)
        progress_noise = torch.randn(len(shape_names), config.sample_points, 2, generator=generator).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: learning_rate_factor(step, config))
    criterion = nn.MSELoss()
    # Keep validation inputs fixed so changes reflect the model, not new samples.
    validation = flow_batch(4096, config, device)
    with torch.no_grad():
        initial_loss = criterion(model(*validation[:3]), validation[3]).item()
    history = []
    window_loss = 0.0
    window_count = 0
    synchronize_device(device)
    started = time.perf_counter()
    print(f"model={model_config.model_type}, device={device}, parameters={sum(p.numel() for p in model.parameters()):,}, initial_val={initial_loss:.5f}", flush=True)

    for step in range(1, config.steps + 1):
        model.train()
        points, times, labels, velocity = flow_batch(config.batch_size, config, device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(points, times, labels), velocity)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip, error_if_nonfinite=True)
        used_lr = optimizer.param_groups[0]["lr"]
        optimizer.step()
        scheduler.step()
        window_loss += loss.item()
        window_count += 1
        if step == 1 or step % config.log_every == 0 or step == config.steps:
            model.eval()
            with torch.no_grad():
                validation_loss = criterion(model(*validation[:3]), validation[3]).item()
            row = {"step": step, "loss": window_loss / window_count, "validation_loss": validation_loss, "learning_rate": used_lr}
            history.append(row)
            print(f"step={step:5d}/{config.steps} loss={row['loss']:.5f} val={validation_loss:.5f} lr={used_lr:.3g}", flush=True)
            window_loss, window_count = 0.0, 0

        if progress_noise is not None and step % config.save_every == 0:
            clouds = []
            for class_id in range(len(shape_names)):
                labels = torch.full((config.sample_points,), class_id, device=device, dtype=torch.long)
                clouds.append(generate_points(model, labels, config.sample_steps, progress_noise[class_id]).cpu())
            image_path = output_dir / "progress" / f"generated_step_{step:06d}.png"
            plot_shapes(clouds, image_path)
            progress_paths.append(image_path)
            print(f"Saved progress: {image_path}", flush=True)

    synchronize_device(device)
    elapsed = time.perf_counter() - started
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "step": config.steps, "config": settings}, output_dir / "checkpoint.pt")
    with (output_dir / "history.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    plot_history(history, output_dir / "loss.png")

    # Re-seed evaluation so generated galleries are repeatable for a checkpoint.
    torch.manual_seed(config.seed + 1)
    generated, targets, metrics = [], [], {}
    trajectories = {}
    for class_id, name in enumerate(shape_names):
        labels = torch.full((config.sample_points,), class_id, device=device, dtype=torch.long)
        recorder = (trajectory_recorder(trajectories, class_id, config.animation_points)
                    if config.inference_animation and class_id % 10 == 9 else None)
        cloud = generate_points(model, labels, config.sample_steps, on_step=recorder)
        target = sample_shape(class_id, config.sample_points, config.noise_std, "cpu").to(device)
        generated.append(cloud.cpu())
        targets.append(target.cpu())
        metrics[name] = {"generated_sliced_w1": sliced_wasserstein(cloud, target), "gaussian_sliced_w1": sliced_wasserstein(torch.randn_like(target), target)}
        print(f"sampled {name}: sliced_w1={metrics[name]['generated_sliced_w1']:.4f}", flush=True)
    plot_shapes(generated, output_dir / "generated.png")
    plot_shapes(generated, output_dir / "comparison.png", targets)
    torch.save({"generated": torch.stack(generated), "targets": torch.stack(targets), "shape_names": shape_names,
                "dataset_version": DATASET_VERSION}, output_dir / "points.pt")
    report = {"initial_validation_loss": initial_loss, "final_validation_loss": history[-1]["validation_loss"], "training_seconds": elapsed, "metrics": metrics}
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    save_inference_animation(trajectories, shape_names, output_dir, config.inference_gif_duration_ms, DATASET_VERSION)
    quality = evaluate_clouds(torch.stack(generated), torch.stack(targets), config.noise_std, output_dir)
    print(f"Fit quality: {quality['status']}; {quality.get('passed_classes', 0)}/{len(shape_names)} classes pass", flush=True)
    if progress_paths:
        animation_path = output_dir / "progress.gif"
        build_progress_gif(progress_paths, animation_path, config.gif_duration_ms)
        print(f"Saved animation: {animation_path}", flush=True)
    elif config.save_progress:
        print("No progress GIF: save_every exceeds training steps; no progress frames were saved.", flush=True)
    print(f"Completed in {elapsed:.1f}s training time. Results: {output_dir.resolve()}", flush=True)
