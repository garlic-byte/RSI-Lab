"""Contract tests for interchangeable pointwise velocity models."""

import unittest

import torch

from config.train_config import ModelConfig
from model.registry import MODEL_REGISTRY, build_model
from model.velocity_mlp import VelocityMLP
from model.velocity_class_conditioned_mlp import ClassConditionedMLPBlock
from train.sampling import generate_points


class ModelRegistryTest(unittest.TestCase):
    def test_modulated_global_mlp_preserves_initial_function_and_rng(self):
        config = ModelConfig(model_type="global_mlp", global_mlp_dim=16,
                             global_mlp_bottleneck=8, hidden_dim=32, num_layers=2)
        torch.manual_seed(42)
        baseline = build_model(config)
        baseline_rng = torch.get_rng_state()
        config.model_type = "modulated_global_mlp"
        torch.manual_seed(42)
        candidate = build_model(config)
        self.assertTrue(torch.equal(baseline_rng, torch.get_rng_state()))
        for name, value in baseline.state_dict().items():
            torch.testing.assert_close(candidate.state_dict()[name], value, rtol=0, atol=0)
        points, times, labels = torch.randn(8, 2), torch.rand(8, 1), torch.arange(8)
        actual = candidate(points, times, labels)
        torch.testing.assert_close(actual, baseline(points, times, labels), rtol=0, atol=0)
        actual.square().mean().backward()
        # Zero initialization must still allow the new branch to learn.
        for block in candidate.blocks:
            self.assertGreater(block.modulation[-1].weight.grad.abs().sum().item(), 0.)
        torch.optim.SGD(candidate.parameters(), lr=0.01).step()
        with torch.no_grad():
            self.assertFalse(torch.allclose(candidate(points, times, labels), actual))

    def test_separate_class_condition_changes_output_and_receives_gradients(self):
        torch.manual_seed(42)
        config = ModelConfig(model_type="class_conditioned_mlp", global_mlp_dim=16,
                             global_mlp_bottleneck=8, hidden_dim=32)
        block = ClassConditionedMLPBlock(config)
        features = torch.randn(4, 2, 16)
        condition = torch.randn(4, 16, requires_grad=True)
        output = block(features, condition)
        self.assertFalse(torch.allclose(output, block(features, torch.zeros_like(condition))))
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(condition.grad).all())
        self.assertGreater(condition.grad.abs().sum().item(), 0.)

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
