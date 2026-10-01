"""Contract tests for interchangeable pointwise velocity models."""

import unittest

import torch

from config.train_config import ModelConfig
from model.registry import MODEL_REGISTRY, build_model
from model.velocity_mlp import VelocityMLP
from model.velocity_class_conditioned_mlp import ClassConditionedMLPBlock
from train.sampling import generate_points


class ModelRegistryTest(unittest.TestCase):
    def test_dispersion_guidance_initial_equivalence_and_gradient(self):
        config = ModelConfig(model_type="guided_mixture_mlp", hidden_dim=32, num_layers=2,
                             global_mlp_dim=16, global_mlp_bottleneck=8, mixture_components=8)
        torch.manual_seed(42)
        baseline = build_model(config)
        state = torch.get_rng_state()
        config.model_type = "dispersion_guided_mixture_mlp"
        torch.manual_seed(42)
        candidate = build_model(config)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        x, t, labels = torch.randn(8, 2), torch.rand(8, 1), torch.arange(8)
        torch.testing.assert_close(candidate(x, t, labels), baseline(x, t, labels), rtol=0, atol=0)
        candidate(x, t, labels).square().mean().backward()
        self.assertGreater(candidate.dispersion_projection.weight.grad.abs().sum().item(), 0)

    def test_block_guidance_preserves_initial_function_and_learns(self):
        config = ModelConfig(model_type="guided_mixture_mlp", hidden_dim=32, num_layers=2,
                             global_mlp_dim=16, global_mlp_bottleneck=8, mixture_components=8)
        torch.manual_seed(42)
        baseline = build_model(config)
        state = torch.get_rng_state()
        config.model_type = "block_guided_mixture_mlp"
        torch.manual_seed(42)
        candidate = build_model(config)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        x, t, labels = torch.randn(8, 2), torch.rand(8, 1), torch.arange(8)
        torch.testing.assert_close(candidate(x, t, labels), baseline(x, t, labels), rtol=0, atol=0)
        # Modulation is also zero-initialized: its first update opens the
        # gradient path to the new conditioning projection on the next step.
        optimizer = torch.optim.SGD(candidate.parameters(), lr=0.01)
        candidate(x, t, labels).square().mean().backward()
        optimizer.step()
        optimizer.zero_grad()
        candidate(x, t, labels).square().mean().backward()
        self.assertGreater(candidate.velocity_condition_projection.weight.grad.abs().sum().item(), 0)

    def test_guidance_preserves_initial_function_and_rng(self):
        config = ModelConfig(model_type="oriented_mixture_mlp", hidden_dim=32, num_layers=2,
                             global_mlp_dim=16, global_mlp_bottleneck=8, mixture_components=8)
        torch.manual_seed(42)
        baseline = build_model(config)
        state = torch.get_rng_state()
        config.model_type = "guided_mixture_mlp"
        torch.manual_seed(42)
        candidate = build_model(config)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        x, t, labels = torch.randn(8, 2), torch.rand(8, 1), torch.arange(8)
        torch.testing.assert_close(candidate(x, t, labels), baseline(x, t, labels), rtol=0, atol=0)
        candidate(x, t, labels).square().mean().backward()
        self.assertGreater(candidate.velocity_projection.weight.grad.abs().sum().item(), 0)

    def test_oriented_covariance_matches_rotation(self):
        model = build_model(ModelConfig(model_type="oriented_mixture_mlp", mixture_components=8))
        labels = torch.tensor([0, 12])
        angles = model.component_angles[labels]
        rotation = torch.stack((angles.cos(), -angles.sin(), angles.sin(), angles.cos()), -1).reshape(2, 8, 2, 2)
        eigenvalues = torch.diag_embed((2 * model.component_log_scales[labels]).exp())
        expected = rotation @ eigenvalues @ rotation.transpose(-1, -2)
        diagonal, cross = model.target_covariance(labels)
        torch.testing.assert_close(diagonal, expected.diagonal(dim1=-2, dim2=-1))
        torch.testing.assert_close(cross, expected[..., 0, 1])

    def test_correlated_basis_matches_matrix_gaussian_reference(self):
        config = ModelConfig(model_type="correlated_mixture_mlp", hidden_dim=32, num_layers=2,
                             global_mlp_dim=16, global_mlp_bottleneck=8, mixture_components=4)
        model = build_model(config).double()
        with torch.no_grad():
            model.component_correlations.uniform_(-2, 2)
            model.component_log_scales.uniform_(-3, 0)
            model.component_logits.normal_()
        x, t, labels = torch.randn(8, 2, dtype=torch.float64), torch.linspace(0, 1, 8, dtype=torch.float64)[:, None], torch.arange(8)
        means = model.component_means[labels]
        scales = model.component_log_scales[labels].exp()
        sigma = torch.diag_embed(scales.square())
        cross = 0.99 * model.component_correlations[labels].tanh() * scales.prod(-1)
        sigma[..., 0, 1] = cross
        sigma[..., 1, 0] = cross
        tau = t[:, None, :, None]
        identity = torch.eye(2, dtype=torch.float64)
        variance = (1 - tau).square() * identity + tau.square() * sigma
        distribution = torch.distributions.MultivariateNormal(t[:, None, :] * means, covariance_matrix=variance)
        responsibilities = (distribution.log_prob(x[:, None, :]) + model.component_logits[labels].log_softmax(-1)).softmax(-1)
        solved = torch.linalg.solve(variance, (x[:, None, :] - t[:, None, :] * means)[..., None])
        conditional = means + ((tau * sigma - (1 - tau) * identity) @ solved).squeeze(-1)
        expected = (responsibilities[..., None] * conditional).sum(1)
        torch.testing.assert_close(model.mixture_velocity(x, t, labels), expected)

    def test_gaussian_basis_matches_known_velocity_and_endpoints(self):
        config = ModelConfig(model_type="mixture_mlp", hidden_dim=32, num_layers=2,
                             global_mlp_dim=16, global_mlp_bottleneck=8, mixture_components=4)
        model = build_model(config)
        points, times, labels = torch.randn(8, 2), torch.linspace(0, 1, 8)[:, None], torch.arange(8)
        with torch.no_grad():
            model.component_means.zero_()
            model.component_log_scales.zero_()
        # Independent standard normal endpoints have a known linear field.
        factor = (2 * times - 1) / ((1 - times).square() + times.square())
        torch.testing.assert_close(model.mixture_velocity(points, times, labels), factor * points)
        torch.testing.assert_close(model(points, torch.ones_like(times), labels), points)
        torch.testing.assert_close(model(points, torch.zeros_like(times), labels), -points)

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
