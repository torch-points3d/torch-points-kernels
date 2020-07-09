import torch
import os
import sys
import numpy.testing as npt
import numpy as np
from sklearn.neighbors import KDTree
import unittest
import time

ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.insert(0, ROOT)

from torch_points_kernels import ball_query


class TestRadiusSpeed(unittest.TestCase):
    def test_speed(self):
        start = time.time()
        a = torch.randn(50000, 3).to(torch.float)
        b = torch.randn(10000, 3).to(torch.float)
        batch_a = torch.tensor([0 for i in range(a.shape[0] // 2)] + [1 for i in range(a.shape[0] // 2, a.shape[0])])
        batch_b = torch.tensor([0 for i in range(b.shape[0] // 2)] + [1 for i in range(b.shape[0] // 2, b.shape[0])])
        R = 1
        samples = 50

        idx, dist = ball_query(R, samples, a, b, mode="PARTIAL_DENSE", batch_x=batch_a, batch_y=batch_b, sort=True,)
        idx1, dist = ball_query(R, samples, a, b, mode="PARTIAL_DENSE", batch_x=batch_a, batch_y=batch_b, sort=True,)
        print(time.time() - start)
        torch.testing.assert_allclose(idx1, idx)

        self.assertEqual(idx.shape[0], b.shape[0])
        self.assertEqual(dist.shape[0], b.shape[0])
        self.assertLessEqual(idx.max().item(), len(batch_a))

        # # Comparison to see if we have the same result
        # tree = KDTree(a.detach().numpy())
        # idx3_sk = tree.query_radius(b.detach().numpy(), r=R)
        # i = np.random.randint(len(batch_b))
        # for p in idx[i].detach().numpy():
        #     if p >= 0 and p < len(batch_a):
        #         assert p in idx3_sk[i]


if __name__ == "__main__":
    unittest.main()
