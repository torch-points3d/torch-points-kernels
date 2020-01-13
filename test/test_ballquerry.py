import unittest
import torch
from torch_points import ball_query
import numpy.testing as npt
import numpy as np

from . import run_if_cuda


class TestBall(unittest.TestCase):
    @run_if_cuda
    def test_simple_gpu(self):
        a = torch.tensor([[[0, 0, 0], [1, 0, 0], [2, 0, 0]]]).to(torch.float).cuda()
        b = torch.tensor([[[0, 0, 0]]]).to(torch.float).cuda()

        npt.assert_array_equal(ball_query(1, 2, a, b).detach().cpu().numpy(), np.array([[[0, 0]]]))

    @run_if_cuda
    def test_larger_gpu(self):
        a = torch.randn(32, 4096, 3).to(torch.float).cuda()
        idx = ball_query(1, 64, a, a).detach().cpu().numpy()
        self.assertGreaterEqual(idx.min(), 0)

    @run_if_cuda
    def test_cpu_gpu_equality(self):
        a = torch.randn(5, 1000, 3)
        res_cpu = ball_query(0.1, 17, a, a).detach().numpy()
        res_cuda = ball_query(0.1, 17, a.cuda(), a.cuda()).cpu().detach().numpy()
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                # Because it is not necessary the same order
                assert set(res_cpu[i][j]) == set(res_cuda[i][j])


class TestBallPartial(unittest.TestCase):
    @run_if_cuda
    def test_simple_gpu(self):
        x = torch.tensor([[10, 0, 0], [0.1, 0, 0], [10, 0, 0], [0.1, 0, 0]]).to(torch.float).cuda()
        y = torch.tensor([[0, 0, 0]]).to(torch.float).cuda()
        batch_x = torch.from_numpy(np.asarray([0, 0, 1, 1])).long().cuda()
        batch_y = torch.from_numpy(np.asarray([0])).long().cuda()

        batch_x = torch.from_numpy(np.asarray([0, 0, 1, 1])).long().cuda()
        batch_y = torch.from_numpy(np.asarray([0])).long().cuda()

        idx, dist2 = ball_query(1.0, 2, x, y, mode="PARTIAL_DENSE", batch_x=batch_x, batch_y=batch_y)

        idx = idx.detach().cpu().numpy()
        dist2 = dist2.detach().cpu().numpy()

        idx_answer = np.asarray([[1, 4]])
        dist2_answer = np.asarray([[0.0100, -1.0000]]).astype(np.float32)

        npt.assert_array_almost_equal(idx, idx_answer)
        npt.assert_array_almost_equal(dist2, dist2_answer)

    # def test_simple_cpu(self):
    #     x = torch.tensor([[10, 0, 0], [0.1, 0, 0], [10, 0, 0], [0.1, 0, 0]]).to(torch.float)
    #     y = torch.tensor([[0, 0, 0]]).to(torch.float)

    #     batch_x = torch.from_numpy(np.asarray([0, 0, 1, 1])).long()
    #     batch_y = torch.from_numpy(np.asarray([0])).long()

    #     idx, dist2 = ball_query(1.0, 2, x, y, mode="PARTIAL_DENSE", batch_x=batch_x, batch_y=batch_y)

    #     idx = idx.detach().cpu().numpy()
    #     dist2 = dist2.detach().cpu().numpy()

    #     idx_answer = np.asarray([[1, 1], [0, 1], [1, 1], [1, 1]])
    #     dist2_answer = np.asarray([[-1, -1], [0.01, -1], [-1, -1], [-1, -1]]).astype(np.float32)

    #     npt.assert_array_almost_equal(idx, idx_answer)
    #     npt.assert_array_almost_equal(dist2, dist2_answer)

    def test_random_cpu(self):
        a = torch.randn(1000, 3).to(torch.float)
        b = torch.randn(1500, 3).to(torch.float)
        batch_a = torch.randint(1, (1000,)).sort(0)[0].long()
        batch_b = torch.randint(1, (1500,)).sort(0)[0].long()
        idx, dist = ball_query(1.0, 12, a, b, mode="PARTIAL_DENSE", batch_x=batch_a, batch_y=batch_b)
        idx2, dist2 = ball_query(1.0, 12, b, a, mode="PARTIAL_DENSE", batch_x=batch_b, batch_y=batch_a)


if __name__ == "__main__":
    unittest.main()
