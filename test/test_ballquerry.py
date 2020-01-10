import unittest
import torch
from torch_points import ball_query
import numpy.testing as npt
import numpy as np


class TestBall(unittest.TestCase):
    def test_simple_gpu(self):
        a = torch.tensor([[[0, 0, 0], [1, 0, 0], [2, 0, 0]]]).to(torch.float).cuda()
        b = torch.tensor([[[0, 0, 0]]]).to(torch.float).cuda()

        npt.assert_array_equal(ball_query(1, 2, a, b).detach().cpu().numpy(), np.array([[[0, 0]]]))

    def test_larger_gpu(self):
        a = torch.randn(32, 4096, 3).to(torch.float).cuda()
        idx = ball_query(1, 64, a, a).detach().cpu().numpy()
        self.assertGreaterEqual(idx.min(),0)

    # def test_simple_cpu(self):
    #     a = torch.tensor([[[0, 0, 0], [1, 0, 0], [2, 0, 0]]]).to(torch.float)
    #     b = torch.tensor([[[0, 0, 0]]]).to(torch.float)
    #     npt.assert_array_equal(ball_query(1, 2, a, b).detach().numpy(), np.array([[[0, 0]]]))

    # def test_cpu_gpu_equality(self):
    #     a = torch.randn(5, 1000, 3)
    #     npt.assert_array_equal(ball_query(0.1, 17, a, a).detach().numpy(),
    #                            ball_query(0.1, 17, a.cuda(), a.cuda()).cpu().detach().numpy())

    def test_partial_gpu(self):
        x = torch.tensor([[10, 0, 0], [0.1, 0, 0], [10, 0, 0], [0.1, 0, 0]]).to(torch.float).cuda()
        y = torch.tensor([[0, 0, 0]]).to(torch.float).cuda()
        batch_x = torch.from_numpy(np.asarray([0, 0, 1, 1])).long().cuda()
        batch_y = torch.from_numpy(np.asarray([0])).long().cuda()
        
        batch_x = torch.from_numpy(np.asarray([0, 0, 1, 1])).long().cuda()
        batch_y = torch.from_numpy(np.asarray([0])).long().cuda()

        idx, dist2 = ball_query(1., 2, x, y, mode="PARTIAL_DENSE", batch_x=batch_x, batch_y=batch_y)

        idx = idx.detach().cpu().numpy()
        dist2 = dist2.detach().cpu().numpy()

        idx_answer = np.asarray([[1, 4]])
        dist2_answer = np.asarray([[ 0.0100, -1.0000]]).astype(np.float32)

        npt.assert_array_almost_equal(idx, idx_answer)
        npt.assert_array_almost_equal(dist2, dist2_answer)


if __name__ == "__main__":
    unittest.main()
