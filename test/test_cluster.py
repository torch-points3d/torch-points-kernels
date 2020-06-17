import unittest
import torch
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.insert(0, ROOT)

from torch_points_kernels.cluster import grow_proximity, region_grow


class TestGrow(unittest.TestCase):
    def setUp(self):
        self.pos = torch.tensor(
            [
                [0, 0, 0],
                [1, 0, 0],
                [2, 0, 0],
                [10, 0, 0],
                [0, 0, 0],
                [1, 0, 0],
                [2, 0, 0],
                [10, 0, 0],
            ]
        )
        self.batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        self.labels = torch.tensor([0, 0, 1, 1, 0, 1, 1, 10])

    def test_simple(self):
        clusters = grow_proximity(self.pos, self.batch, radius=2, min_cluster_size=1)
        self.assertEqual(clusters, [[0, 1, 2], [3], [4, 5, 6], [7]])

        clusters = grow_proximity(self.pos, self.batch, radius=2, min_cluster_size=3)
        self.assertEqual(clusters, [[0, 1, 2], [4, 5, 6]])

    def test_region_grow(self):
        clusters = region_grow(
            self.pos, self.labels, self.batch, radius=2, min_cluster_size=1
        )
        self.assertEqual(len(clusters[0]), 2)
        self.assertEqual(len(clusters[1]), 3)
        self.assertEqual(len(clusters[10]), 1)
        torch.testing.assert_allclose(clusters[0][0], torch.tensor([0, 1]))
        torch.testing.assert_allclose(clusters[0][1], torch.tensor([4]))
        torch.testing.assert_allclose(clusters[1][0], torch.tensor([2]))
        torch.testing.assert_allclose(clusters[1][1], torch.tensor([3]))
        torch.testing.assert_allclose(clusters[1][2], torch.tensor([5, 6]))
        torch.testing.assert_allclose(clusters[10][0], torch.tensor([7]))


if __name__ == "__main__":
    unittest.main()
