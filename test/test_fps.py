import unittest
import torch
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.insert(0, ROOT)

from torch_points_kernels.points_cpu import fps


class TestFps(unittest.TestCase):
    def test_simplecpu(self):
        points = torch.tensor([[[0, 0, 0], [1, 0, 0], [2, 0, 0]], [[-1, 1, 0], [0, 0, 10], [0, 0, 2]]]).float()
        idx = fps(points, 2, False)
        torch.testing.assert_allclose(idx, torch.tensor([[0, 2], [0, 1]]))

    def test_random(self):
        points = torch.randn(10, 100, 3)
        idx = fps(points, 2, True)
        self.assertNotEqual(idx[0][0], 0)


if __name__ == "__main__":
    unittest.main()
