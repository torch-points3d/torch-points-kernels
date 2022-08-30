import unittest
import torch
import os
import sys



ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.insert(0, ROOT)

from torch_points_kernels.points_cpu import fps
from torch_points_kernels import furthest_point_sample
from test import run_if_cuda


class TestFps(unittest.TestCase):
    def test_simplecpu(self):
        points = torch.tensor([[[0, 0, 0], [1, 0, 0], [2, 0, 0]], [[-1, 1, 0], [0, 0, 10], [0, 0, 2]]]).float()
        idx = fps(points, 2, False)
        torch.testing.assert_allclose(idx, torch.tensor([[0, 2], [0, 1]]))

    def test_random(self):
        points = torch.randn(10, 100, 3)
        idx = fps(points, 2, True)
        self.assertNotEqual(idx[0][0], 0)

    @run_if_cuda
    def test_gpu(self):
        points = torch.randn([16, 100, 3]).cuda()
        nsamples = 2
        idx = furthest_point_sample(points,nsamples)
        idx_cpu = furthest_point_sample(points.cpu(),nsamples)
        sorted_idx, _ = torch.sort(idx.cpu())
        sorted_idx_cpu, _ = torch.sort(idx_cpu)
        torch.testing.assert_allclose(sorted_idx,sorted_idx_cpu)

if __name__ == "__main__":
    unittest.main()
