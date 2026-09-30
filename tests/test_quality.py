"""Ensure diagnostics detect blur, missing support and collapsed predictions."""

import unittest

import torch

from data.shapes import LEGACY_VERSION, sample_shape
from evaluation.quality import cloud_metrics, grid_probabilities


class QualityTest(unittest.TestCase):
    def test_shape_failures(self):
        torch.manual_seed(9)
        reference = sample_shape(4, 1000, dataset_version=LEGACY_VERSION)
        independent = sample_shape(4, 1000, dataset_version=LEGACY_VERSION)
        radius = torch.quantile(torch.cdist(independent, reference).min(1).values, .95).item()
        real = cloud_metrics(independent, reference, radius)
        collapsed = cloud_metrics(torch.zeros_like(reference), reference, radius)
        blurred = cloud_metrics(reference + torch.randn_like(reference) * .3, reference, radius)
        half = reference[reference[:, 0] > 0].repeat(3, 1)[:1000]
        missing = cloud_metrics(half, reference, radius)
        self.assertGreater(real["precision"], .9)
        self.assertLess(collapsed["coverage"], .1)
        self.assertGreater(blurred["grid_js"], 2 * real["grid_js"])
        self.assertLess(blurred["precision"], .5)
        self.assertGreater(missing["precision"], .95)
        self.assertLess(missing["coverage"], .65)

    def test_out_of_bounds_mass_is_not_discarded(self):
        probabilities = grid_probabilities(torch.tensor([[0., 0.], [5., 5.]]), 32)
        self.assertAlmostEqual(probabilities.sum().item(), 1.)
        self.assertAlmostEqual(probabilities[-1].item(), .5)


if __name__ == "__main__":
    unittest.main()
