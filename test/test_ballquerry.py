import unittest
import torch
from torch_points import ball_query
import numpy.testing as npt
import numpy as np
from sklearn.neighbors import KDTree

from . import run_if_cuda


class TestBall(unittest.TestCase):
    @run_if_cuda
    def test_simple_gpu(self):
        a = torch.tensor([[[0, 0, 0], [1, 0, 0], [2, 0, 0]]]).to(torch.float).cuda()
        b = torch.tensor([[[0, 0, 0]]]).to(torch.float).cuda()

        npt.assert_array_equal(ball_query(0.5, 2, a, b).detach().cpu().numpy(), np.array([[[0, 0]]]))

    def test_simple_cpu(self):
        a = torch.tensor([[[0, 0, 0], [1, 0, 0], [2, 0, 0]], [[0, 0, 0], [1, 0, 0], [2, 0, 0]]]).to(torch.float)
        b = torch.tensor([[[0, 0, 0]], [[-1, 0, 0]]]).to(torch.float)

        npt.assert_array_equal(ball_query(0.5, 2, a, b).detach().cpu().numpy(), np.array([[[0, 0]], [[0, 0]]]))

    @run_if_cuda
    def test_larger_gpu(self):
        a = torch.randn(32, 4096, 3).to(torch.float).cuda()
        idx = ball_query(1, 64, a, a).detach().cpu().numpy()
        self.assertGreaterEqual(idx.min(), 0)

    @run_if_cuda
    def test_cpu_gpu_equality(self):
        a = torch.randn(2, 10, 3)
        b = torch.randn(2, 5, 3)
        res_cpu = ball_query(1, 17, a, b).detach().numpy()
        res_cuda = ball_query(1, 17, a.cuda(), b.cuda()).cpu().detach().numpy()
        for i in range(b.shape[0]):
            for j in range(b.shape[1]):
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

    def test_simple_cpu(self):
        x = torch.tensor([[10, 0, 0], [0.1, 0, 0], [10, 0, 0], [0.1, 0, 0]]).to(torch.float)
        y = torch.tensor([[0, 0, 0]]).to(torch.float)

        batch_x = torch.from_numpy(np.asarray([0, 0, 1, 1])).long()
        batch_y = torch.from_numpy(np.asarray([0])).long()

        idx, dist2 = ball_query(1.0, 2, x, y, mode="PARTIAL_DENSE", batch_x=batch_x, batch_y=batch_y)

        idx = idx.detach().cpu().numpy()
        dist2 = dist2.detach().cpu().numpy()

        idx_answer = np.asarray([[1, 4]])
        dist2_answer = np.asarray([[0.0100, -1.0000]]).astype(np.float32)

        npt.assert_array_almost_equal(idx, idx_answer)
        npt.assert_array_almost_equal(dist2, dist2_answer)

    def test_random_cpu(self):
        a = torch.randn(100, 3).to(torch.float)
        b = torch.randn(50, 3).to(torch.float)
        batch_a = torch.zeros((a.shape[0],)).long()
        batch_b = torch.zeros((b.shape[0],)).long()

        idx, dist = ball_query(0.5, 17, a, b, mode="PARTIAL_DENSE", batch_x=batch_a, batch_y=batch_b)
        self.assertEqual(idx.shape[0], b.shape[0])
        self.assertEqual(dist.shape[0], b.shape[0])

        # Comparison to see if we have the same result
        tree = KDTree(a.detach().numpy())
        idx3_sk = tree.query_radius(b.detach().numpy(), r=0.5)
        i = np.random.randint(len(batch_b))
        for p in idx[i].detach().numpy():
            if p < len(batch_b):
                assert p in idx3_sk[i]


if __name__ == "__main__":
    unittest.main()
