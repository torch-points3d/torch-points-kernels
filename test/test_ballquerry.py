import unittest
import torch
from torch_points import ball_query_dense
import numpy.testing as npt
import numpy as np


class TestBall(unittest.TestCase):
    def test_simple_gpu(self):
        a = torch.tensor([[[0, 0, 0], [1, 0, 0], [2, 0, 0]]]).to(torch.float).cuda()
        b = torch.tensor([[[0, 0, 0]]]).to(torch.float).cuda()

        npt.assert_array_equal(ball_query_dense(1, 2, a, b).detach().cpu().numpy(), np.array([[[0, 0]]]))

    def test_simple_cpu(self):
        a = torch.tensor([[[0, 0, 0], [1, 0, 0], [2, 0, 0]]]).to(torch.float)
        b = torch.tensor([[[0, 0, 0]]]).to(torch.float)
        npt.assert_array_equal(ball_query_dense(1, 2, a, b).detach().numpy(), np.array([[[0, 0]]]))

    def test_cpu_gpu_equality(self):
        a = torch.randn(5, 1000, 3)
        res_cpu = ball_query_dense(0.1, 17, a, a).detach().numpy()
        res_cuda = ball_query_dense(0.1, 17, a.cuda(), a.cuda()).cpu().detach().numpy()
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                # Because it is not necessary the same order
                assert set(res_cpu[i][j]) == set(res_cuda[i][j])


if __name__ == "__main__":
    unittest.main()
