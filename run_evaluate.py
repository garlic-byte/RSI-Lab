"""Evaluate existing generated points without retraining the model."""

import argparse
import json
from dataclasses import fields
from pathlib import Path

import torch

from config.evaluation_config import EvaluationConfig
from evaluation.quality import evaluate_clouds
from data.shapes import LEGACY_VERSION


def main() -> None:
    """Read an experiment's saved points and target noise configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    defaults = EvaluationConfig()
    for field in fields(defaults):
        value = getattr(defaults, field.name)
        parser.add_argument("--" + field.name.replace("_", "-"), type=type(value), default=value)
    args = vars(parser.parse_args())
    directory = args.pop("output_dir")
    torch.set_num_threads(4)
    settings = json.loads((directory / "config.json").read_text())
    points = torch.load(directory / "points.pt", map_location="cpu", weights_only=True)
    dataset_version = settings.get("dataset_version", LEGACY_VERSION)
    if points.get("dataset_version", dataset_version) != dataset_version:
        raise ValueError("Point data and config have different dataset versions")
    report = evaluate_clouds(points["generated"], points["targets"], settings["train"]["noise_std"], directory,
                             EvaluationConfig(**args), dataset_version=dataset_version)
    print(f"Quality: {report['status']}; passed {report.get('passed_classes', 0)}/{report.get('total_classes', 0)} classes")
    print(f"Report: {(directory / 'quality_report.json').resolve()}")


if __name__ == "__main__":
    main()
