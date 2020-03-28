import unittest
import torch
from torch_points_kernels import ball_query
import numpy.testing as npt
import numpy as np
from sklearn.neighbors import KDTree

from . import run_if_cuda


class TestBall(unittest.TestCase):
    @run_if_cuda
    def test_simple_gpu(self):
        a = torch.tensor([[[0, 0, 0], [1, 0, 0], [2, 0, 0]], [[0, 0, 0], [1, 0, 0], [2, 0, 0]]]).to(torch.float).cuda()
        b = torch.tensor([[[0, 0, 0]], [[3, 0, 0]]]).to(torch.float).cuda()
        idx, dist = ball_query(1.01, 2, a, b)
        torch.testing.assert_allclose(idx.cpu(), torch.tensor([[[0, 1]], [[2, 2]]]))
        torch.testing.assert_allclose(dist.cpu(), torch.tensor([[[0, 1]], [[1, -1]]]).float())

    def test_simple_cpu(self):
        a = torch.tensor([[[0, 0, 0], [1, 0, 0], [2, 0, 0]], [[0, 0, 0], [1, 0, 0], [2, 0, 0]]]).to(torch.float)
        b = torch.tensor([[[0, 0, 0]], [[3, 0, 0]]]).to(torch.float)
        idx, dist = ball_query(1.01, 2, a, b, sort=True)
        torch.testing.assert_allclose(idx, torch.tensor([[[0, 1]], [[2, 2]]]))
        torch.testing.assert_allclose(dist, torch.tensor([[[0, 1]], [[1, -1]]]).float())

        a = torch.tensor([[[0, 0, 0], [1, 0, 0], [1, 1, 0]]]).to(torch.float)
        idx, dist = ball_query(1.01, 3, a, a, sort=True)
        torch.testing.assert_allclose(idx, torch.tensor([[[0, 1, 0], [1, 0, 2], [2, 1, 2]]]))

    @run_if_cuda
    def test_larger_gpu(self):
        a = torch.randn(32, 4096, 3).to(torch.float).cuda()
        idx, dist = ball_query(1, 64, a, a)
        self.assertGreaterEqual(idx.min(), 0)

    @run_if_cuda
    def test_cpu_gpu_equality(self):
        a = torch.randn(5, 1000, 3)
        b = torch.randn(5, 500, 3)
        res_cpu = ball_query(1, 500, a, b)[0].detach().numpy()
        res_cuda = ball_query(1, 500, a.cuda(), b.cuda())[0].cpu().detach().numpy()
        for i in range(b.shape[0]):
            for j in range(b.shape[1]):
                # Because it is not necessary the same order
                assert set(res_cpu[i][j]) == set(res_cuda[i][j])

        res_cpu = ball_query(0.01, 500, a, b)[0].detach().numpy()
        res_cuda = ball_query(0.01, 500, a.cuda(), b.cuda())[0].cpu().detach().numpy()
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

        idx_answer = np.asarray([[1, -1]])
        dist2_answer = np.asarray([[0.100, -1.0000]]).astype(np.float32)

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

        idx_answer = np.asarray([[1, -1]])
        dist2_answer = np.asarray([[0.100, -1.0000]]).astype(np.float32)

        npt.assert_array_almost_equal(idx, idx_answer)
        npt.assert_array_almost_equal(dist2, dist2_answer)

    def test_random_cpu(self):
        a = torch.randn(100, 3).to(torch.float)
        b = torch.randn(50, 3).to(torch.float)
        batch_a = torch.tensor([0 for i in range(a.shape[0] // 2)] + [1 for i in range(a.shape[0] // 2, a.shape[0])])
        batch_b = torch.tensor([0 for i in range(b.shape[0] // 2)] + [1 for i in range(b.shape[0] // 2, b.shape[0])])
        R = 1

        idx, dist = ball_query(R, 15, a, b, mode="PARTIAL_DENSE", batch_x=batch_a, batch_y=batch_b, sort=True)
        idx1, dist = ball_query(R, 15, a, b, mode="PARTIAL_DENSE", batch_x=batch_a, batch_y=batch_b, sort=True)
        torch.testing.assert_allclose(idx1, idx)
        with self.assertRaises(AssertionError):
            idx, dist = ball_query(R, 15, a, b, mode="PARTIAL_DENSE", batch_x=batch_a, batch_y=batch_b, sort=False)
            idx1, dist = ball_query(R, 15, a, b, mode="PARTIAL_DENSE", batch_x=batch_a, batch_y=batch_b, sort=False)
            torch.testing.assert_allclose(idx1, idx)

        self.assertEqual(idx.shape[0], b.shape[0])
        self.assertEqual(dist.shape[0], b.shape[0])
        self.assertLessEqual(idx.max().item(), len(batch_a))

        # Comparison to see if we have the same result
        tree = KDTree(a.detach().numpy())
        idx3_sk = tree.query_radius(b.detach().numpy(), r=R)
        i = np.random.randint(len(batch_b))
        for p in idx[i].detach().numpy():
            if p >= 0 and p < len(batch_a):
                assert p in idx3_sk[i]


if __name__ == "__main__":
    unittest.main()
