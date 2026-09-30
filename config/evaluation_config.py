"""Explicit engineering acceptance tolerances for point-cloud fitting."""

from dataclasses import dataclass


@dataclass
class EvaluationConfig:
    """Calibrate against independent draws from the known target sampler."""

    max_points: int = 1000
    calibration_trials: int = 5
    distance_tolerance: float = 1.5
    """Allow distance metrics up to 1.5 times the calibration 95th percentile."""
    probability_slack: float = 0.05
    """Allow precision/coverage five percentage points below calibration p05."""
    seed: int = 2026

    def validate(self) -> None:
        if self.max_points < 256 or self.calibration_trials < 3:
            raise ValueError("Evaluation needs max_points >= 256 and calibration_trials >= 3")
        if self.distance_tolerance < 1 or not 0 <= self.probability_slack < 1:
            raise ValueError("distance_tolerance must be >= 1 and probability_slack in [0,1)")
