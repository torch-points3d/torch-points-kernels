import unittest
import torch
from torch_points import knn

class TestKnn(unittest.TestCase):
    def test_cpu(self):
        support = torch.tensor([[[0,0,0],[1,0,0],[2,0,0]]])
        query = torch.tensor([[[0,0,0]]])

        idx, dist = knn(support, query, 3)
        torch.testing.assert_allclose(idx, torch.tensor([[[0,1, 2]]]))
        torch.testing.assert_allclose(dist, torch.tensor([[[0.,1., 4.]]]))

        idx, dist = knn(support, query, 2)
        torch.testing.assert_allclose(idx, torch.tensor([[[0,1]]]))

        with self.assertRaises(RuntimeError):
            knn(support,query, 5)