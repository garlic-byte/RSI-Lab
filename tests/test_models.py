"""Contract tests for interchangeable pointwise velocity models."""

import unittest

import torch

from config.train_config import ModelConfig
from model.registry import MODEL_REGISTRY, build_model
from model.velocity_mlp import VelocityMLP
from train.sampling import generate_points


class ModelRegistryTest(unittest.TestCase):
    def test_forward_backward_sampling_and_batch_independence(self):
        torch.set_num_threads(4)
        for name in MODEL_REGISTRY:
            with self.subTest(model=name):
                config = ModelConfig(model_type=name, hidden_dim=32, num_layers=2, transformer_dim=16)
                model = build_model(config)
                points, time, labels = torch.randn(8, 2), torch.rand(8, 1), torch.tensor([0, 9, 19, 39, 49, 59, 89, 99])
                velocity = model(points, time, labels)
                self.assertEqual(velocity.shape, (8, 2))
                velocity.square().mean().backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
                model.eval()
                with torch.no_grad():
                    together = model(points, time, labels)
                    separate = torch.cat([model(points[i:i+1], time[i:i+1], labels[i:i+1]) for i in range(8)])
                    torch.testing.assert_close(together, separate, atol=1e-6, rtol=1e-5)
                samples = generate_points(model, labels, steps=3)
                self.assertTrue(torch.isfinite(samples).all())
                copy = build_model(config)
                copy.load_state_dict(model.state_dict())
                torch.testing.assert_close(copy(points, time, labels), together)

    def test_legacy_configuration_and_bad_architectures(self):
        self.assertIsInstance(build_model(ModelConfig(**{"hidden_dim": 32, "num_layers": 2})), VelocityMLP)
        with self.assertRaises(ValueError):
            build_model(ModelConfig(model_type="missing"))
        with self.assertRaises(ValueError):
            build_model(ModelConfig(model_type="transformer", transformer_dim=63, num_heads=4))


if __name__ == "__main__":
    unittest.main()
