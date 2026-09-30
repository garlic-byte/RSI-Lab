"""Check particle identities, actual integration states and GIF export."""

import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image
from torch import nn

from train.sampling import generate_points
from utils.inference_animation import trajectory_recorder, save_inference_animation


class LinearVelocity(nn.Module):
    """dx/dt=x supplies a known discrete Heun update for every particle."""

    def forward(self, points, time, labels):
        return points


class InferenceAnimationTest(unittest.TestCase):
    def test_trajectory_is_solver_state_and_preserves_rng(self):
        model = LinearVelocity()
        noise = torch.tensor([[.1, .2], [-.3, .4], [.5, -.6]])
        original = noise.clone()
        labels = torch.full((3,), 9, dtype=torch.long)
        paths = {}
        rng = torch.get_rng_state().clone()
        result = generate_points(model, labels, 4, noise, trajectory_recorder(paths, 9, 2))
        self.assertEqual(len(paths[9]), 5)
        factor = 1 + .25 + .5 * .25 ** 2
        for step, positions in enumerate(paths[9]):
            torch.testing.assert_close(positions, noise[:2] * factor ** step)
        torch.testing.assert_close(paths[9][-1], result[:2])
        self.assertTrue(torch.equal(noise, original))
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        self.assertTrue(model.training)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            save_inference_animation(paths, [str(i) for i in range(10)], output, 50)
            with Image.open(output / "inference.gif") as gif:
                self.assertEqual(gif.n_frames, 5)
                self.assertEqual(gif.info["loop"], 0)
            stored = torch.load(output / "inference_trajectories.pt", weights_only=True)
            self.assertEqual(stored["positions"].shape, (1, 5, 2, 2))
            self.assertEqual(stored["class_ids"], [9])


if __name__ == "__main__":
    unittest.main()
