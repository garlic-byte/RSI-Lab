"""Check analytic distributions, flow gradients and scheduler boundaries."""

import unittest

import torch

from config.train_config import ModelConfig, TrainConfig
from data.shapes import SHAPE_NAMES, LEGACY_VERSION, get_shape_names, sample_shape, sample_batch
from model.velocity_mlp import VelocityMLP
from train.sampling import generate_points
from train.trainer import learning_rate_factor


class PipelineTest(unittest.TestCase):
    def test_checkerboard_family(self):
        for class_id in range(10):
            size = class_id + 4
            points = sample_shape(class_id, 12000, noise_std=0.)
            self.assertTrue((points.abs() <= 1.4).all())
            cells = ((points + 1.4) / (2.8 / size)).floor().long()
            self.assertTrue((cells.sum(1) % 2 == 0).all())
            self.assertEqual(len(cells.unique(dim=0)), (size * size + 1) // 2)
            self.assertEqual(SHAPE_NAMES[class_id], f"Checkerboard {size}x{size}")

    def test_legacy_and_unchanged_extended_classes(self):
        self.assertEqual(get_shape_names(LEGACY_VERSION)[4], "Circle")
        for class_id in range(10, 100):
            torch.manual_seed(71)
            current = sample_shape(class_id, 128)
            torch.manual_seed(71)
            legacy = sample_shape(class_id, 128, dataset_version=LEGACY_VERSION)
            self.assertTrue(torch.equal(current, legacy))
        with self.assertRaises(ValueError):
            get_shape_names("unknown")

    def test_distributions(self):
        self.assertEqual(len(SHAPE_NAMES), 100)
        self.assertEqual(len(set(SHAPE_NAMES)), 100)
        for class_id in range(len(SHAPE_NAMES)):
            points = sample_shape(class_id, 128)
            self.assertEqual(points.shape, (128, 2))
            self.assertTrue(torch.isfinite(points).all())
            self.assertGreater(points.std().item(), 0.1)

    def test_backward_and_sampling(self):
        model = VelocityMLP(ModelConfig(hidden_dim=32, num_layers=2))
        labels = torch.arange(100)
        result = model(torch.randn(100, 2), torch.rand(100, 1), labels)
        result.square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        samples = generate_points(model, labels, steps=3)
        self.assertEqual(samples.shape, (100, 2))
        self.assertTrue(torch.isfinite(samples).all())
        self.assertTrue(model.training)

    def test_class_count_sampling(self):
        torch.manual_seed(123)
        points, labels = sample_batch(5000, 0.035, "cpu", 100)
        self.assertEqual(points.shape, (5000, 2))
        self.assertEqual(len(labels.unique()), 100)
        _, old_labels = sample_batch(500, 0.035, "cpu", 10)
        self.assertTrue((old_labels < 10).all())
        with self.assertRaises(ValueError):
            sample_batch(5, 0.035, "cpu", 101)

    def test_scheduler(self):
        config = TrainConfig()
        self.assertAlmostEqual(learning_rate_factor(0, config), 1 / config.warmup_steps)
        self.assertAlmostEqual(learning_rate_factor(config.warmup_steps - 1, config), 1)
        self.assertAlmostEqual(learning_rate_factor(config.steps - 1, config), config.min_lr_ratio)

    def test_fixed_noise_does_not_change_training_rng(self):
        model = VelocityMLP(ModelConfig(hidden_dim=32, num_layers=2))
        labels = torch.arange(10)
        noise = torch.randn(10, 2)
        original = noise.clone()
        rng_state = torch.get_rng_state().clone()
        first = generate_points(model, labels, steps=3, initial_noise=noise)
        second = generate_points(model, labels, steps=3, initial_noise=noise)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(noise, original))
        self.assertTrue(torch.equal(torch.get_rng_state(), rng_state))


if __name__ == "__main__":
    unittest.main()
