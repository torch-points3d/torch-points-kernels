import unittest
import torch
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.insert(0, ROOT)

from torch_points_kernels import three_nn, knn
from . import run_if_cuda


class TestKnn(unittest.TestCase):
    def test_cpu(self):
        support = torch.tensor([[[0, 0, 0], [1, 0, 0], [2, 0, 0]]]).float()
        query = torch.tensor([[[0, 0, 0]]]).float()

        idx, dist = knn(support, query, 3)
        torch.testing.assert_allclose(idx, torch.tensor([[[0, 1, 2]]]))
        torch.testing.assert_allclose(dist, torch.tensor([[[0.0, 1.0, 4.0]]]))

        idx, dist = knn(support, query, 2)
        torch.testing.assert_allclose(idx, torch.tensor([[[0, 1]]]))

        with self.assertRaises(RuntimeError):
            knn(support, query, 5)

    def test_larger_cpu(self):
        support = torch.randn((2, 10, 3))
        query = torch.randn((2, 10, 3))

        idx, dist = knn(support, query, 3)


class TestThreeNN(unittest.TestCase):
    @run_if_cuda
    def test_cpugpu(self):
        b = 20
        n = 1000
        known = torch.randn((b, 2 * n, 3))
        unknown = torch.randn((b, n, 3))

        dist_cpu, cpu_idx = three_nn(unknown, known)
        dist_cuda, cuda_idx = three_nn(unknown.cuda(), known.cuda())

        torch.testing.assert_allclose(dist_cpu, dist_cuda.cpu())


if __name__ == "__main__":
    unittest.main()
