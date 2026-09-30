"""Calibrated point-cloud checks: global distance, support and local density.

These are finite-sample engineering checks, not a statistical equivalence test.
Nearest-distance precision/coverage below are custom fixed-radius diagnostics,
not the published k-nearest-neighbor PRDC metric.
"""

import csv
import json
import math
from dataclasses import asdict
from pathlib import Path

import torch

from config.evaluation_config import EvaluationConfig
from data.shapes import DATASET_VERSION, get_shape_names, sample_shape
from utils.plotting import plt


def sliced_w1(first: torch.Tensor, second: torch.Tensor) -> float:
    """Average absolute quantile distance across 128 1D projections."""
    angles = torch.linspace(0, math.pi, 129)[:-1]
    directions = torch.stack((angles.cos(), angles.sin()))
    return ((first @ directions).sort(0).values - (second @ directions).sort(0).values).abs().mean().item()


def grid_probabilities(points: torch.Tensor, bins: int) -> torch.Tensor:
    """Histogram inside [-1.9,1.9]^2 with an explicit out-of-bounds bin."""
    indices = ((points + 1.9) / 3.8 * bins).floor().long()
    valid = ((indices >= 0) & (indices < bins)).all(1)
    flat = torch.full((len(points),), bins * bins, dtype=torch.long)
    flat[valid] = indices[valid, 1] * bins + indices[valid, 0]
    return torch.bincount(flat, minlength=bins * bins + 1).float() / len(points)


def grid_js(first: torch.Tensor, second: torch.Tensor) -> float:
    """Average Jensen-Shannon divergence (bits) on 32x32 and 64x64 grids."""
    values = []
    for bins in (32, 64):
        p, q = grid_probabilities(first, bins), grid_probabilities(second, bins)
        mean = (p + q) / 2
        def relative_entropy(a: torch.Tensor) -> torch.Tensor:
            mask = a > 0
            return (a[mask] * (a[mask] / mean[mask]).log2()).sum()
        values.append((relative_entropy(p) + relative_entropy(q)) / 2)
    return torch.stack(values).mean().item()


def cloud_metrics(generated: torch.Tensor, reference: torch.Tensor, radius: float) -> dict[str, float]:
    """Report proximity in both directions plus global/local distribution gaps."""
    distances = torch.cdist(generated, reference)
    return {"sliced_w1": sliced_w1(generated, reference), "grid_js": grid_js(generated, reference),
            "precision": (distances.min(1).values <= radius).float().mean().item(),
            "coverage": (distances.min(0).values <= radius).float().mean().item()}


@torch.inference_mode()
def evaluate_clouds(generated: torch.Tensor, targets: torch.Tensor, noise_std: float,
                    output_dir: Path, config: EvaluationConfig | None = None,
                    dataset_version: str = DATASET_VERSION) -> dict:
    """Evaluate every class against its analytic sampler; write CSV/JSON/chart.

    Args:
        generated: Saved predictions [classes,N,2].
        targets: Independent target reference clouds of the same shape.
        noise_std: Target sampler noise, identical to the training configuration.
        output_dir: Directory for quality_report.json, quality.csv and quality.png.
        config: Calibration and acceptance tolerances.
    """
    config = config or EvaluationConfig()
    shape_names = get_shape_names(dataset_version)
    config.validate()
    output_dir.mkdir(parents=True, exist_ok=True)
    if generated.shape != targets.shape or generated.ndim != 3 or generated.shape[-1] != 2:
        raise ValueError("generated and targets must have matching [classes,N,2] shapes")
    if not 1 <= len(generated) <= len(shape_names):
        raise ValueError("Class count must be in the supported catalog range")
    count = min(generated.shape[1], config.max_points)
    if count < 256:
        report = {"status": "insufficient_samples", "points_per_class": count,
                  "reason": "At least 256 points per class required; no pass/fail judgment made"}
        (output_dir / "quality_report.json").write_text(json.dumps(report, indent=2) + "\n")
        return report
    if not torch.isfinite(generated).all() or not torch.isfinite(targets).all():
        raise ValueError("Cannot evaluate non-finite point coordinates")
    rows = []
    generated, targets = generated.cpu(), targets.cpu()
    # Evaluation must not perturb training or sampling random-number streams.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(config.seed)
        for class_id in range(len(generated)):
            selected = torch.randperm(generated.shape[1])[:count]
            cloud, reference = generated[class_id, selected], targets[class_id, selected]
            pilot = sample_shape(class_id, count, noise_std, dataset_version=dataset_version)
            radius = torch.quantile(torch.cdist(pilot, reference).min(1).values, 0.95).item()
            baselines = [cloud_metrics(sample_shape(class_id, count, noise_std, dataset_version=dataset_version), reference, radius)
                         for _ in range(config.calibration_trials)]
            observed = cloud_metrics(cloud, reference, radius)
            thresholds = {}
            failures = []
            severity = []
            for key, value in observed.items():
                samples = torch.tensor([baseline[key] for baseline in baselines])
                if key in ("sliced_w1", "grid_js"):
                    threshold = max(torch.quantile(samples, 0.95).item() * config.distance_tolerance, 1e-8)
                    failed = value > threshold
                    severity.append(value / threshold)
                else:
                    threshold = max(0., torch.quantile(samples, 0.05).item() - config.probability_slack)
                    failed = value < threshold
                    severity.append((1 - value) / max(1 - threshold, 1e-8))
                thresholds[key + "_threshold"] = threshold
                if failed:
                    failures.append(key)
            rows.append({"class_id": class_id, "name": shape_names[class_id], "passed": not failures,
                         "failed_checks": ";".join(failures), "severity": max(severity),
                         "radius": radius, **observed, **thresholds})
    failed = [row["class_id"] for row in rows if not row["passed"]]
    report = {"status": "pass" if not failed else "needs_attention", "dataset_version": dataset_version,
              "passed_classes": len(rows) - len(failed),
              "total_classes": len(rows), "failed_class_ids": failed, "points_per_class": count,
              "config": asdict(config), "rule": "All four checks must pass for each class; all classes must pass overall",
              "limitation": "Heuristic tolerances from finite real-real samples; not proof of equivalence or a calibrated hypothesis test",
              "classes": rows}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "quality_report.json").write_text(json.dumps(report, indent=2) + "\n")
    with (output_dir / "quality.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["severity"], reverse=True))
    # One summary chart replaces manual inspection of a hundred scatter plots.
    fig, ax = plt.subplots(figsize=(16, max(5, len(rows) * 0.16)))
    ordered = sorted(rows, key=lambda row: row["severity"], reverse=True)
    ax.barh(range(len(rows)), [row["severity"] for row in ordered],
            color=["#c7473d" if not row["passed"] else "#32956c" for row in ordered])
    ax.set_yticks(range(len(rows)), [f"{row['class_id']:02d} {row['name']}" for row in ordered], fontsize=7)
    ax.invert_yaxis()
    ax.axvline(1, color="black", linestyle="--")
    ax.set(xlabel="Worst normalized check (<= 1 passes)", title=f"Fit checks: {len(rows)-len(failed)}/{len(rows)} classes pass")
    fig.tight_layout()
    fig.savefig(output_dir / "quality.png", dpi=140)
    plt.close(fig)
    return report
