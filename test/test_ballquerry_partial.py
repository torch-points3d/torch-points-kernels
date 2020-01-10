import unittest
import torch
from torch_points import ball_query
from torch_points.points_cpu import ball_query as cpu_ball_query
from torch_cluster import radius_cuda
import numpy.testing as npt
import numpy as np

class TestBallPartial(unittest.TestCase):
    def test_simple_gpu(self):
        x = torch.tensor([[10, 0, 0], [0.1, 0, 0], [10, 0, 0], [0.1, 0, 0]]).to(torch.float).cuda()
        y = torch.tensor([[0, 0, 0]]).to(torch.float).cuda()
        batch_x = torch.from_numpy(np.asarray([0, 0, 1, 1])).long().cuda()
        batch_y = torch.from_numpy(np.asarray([0])).long().cuda()

        batch_x = torch.from_numpy(np.asarray([0, 0, 1, 1])).long().cuda()
        batch_y = torch.from_numpy(np.asarray([0])).long().cuda()

        idx, dist2 = ball_query(1., 2, x, y, batch_x, batch_y, mode="PARTIAL_DENSE")

        idx = idx.detach().cpu().numpy()
        dist2 = dist2.detach().cpu().numpy()

        idx_answer = np.asarray([[1, 4]])
        dist2_answer = np.asarray([[ 0.0100, -1.0000]]).astype(np.float32)

        npt.assert_array_almost_equal(idx, idx_answer)
        npt.assert_array_almost_equal(dist2, dist2_answer)


    def test_simple_cpu(self):
        x = torch.tensor([[10, 0, 0], [0.1, 0, 0], [10, 0, 0], [0.1, 0, 0]]).to(torch.float)
        y = torch.tensor([[0, 0, 0]]).to(torch.float)
        batch_x = torch.from_numpy(np.asarray([0, 0, 1, 1])).long()
        batch_y = torch.from_numpy(np.asarray([0])).long()

        batch_x = torch.from_numpy(np.asarray([0, 0, 1, 1])).long()
        batch_y = torch.from_numpy(np.asarray([0])).long()

        idx, dist2 = ball_query(1., 2, x, y, batch_x, batch_y, mode="PARTIAL_DENSE")

        idx = idx.detach().cpu().numpy()
        dist2 = dist2.detach().cpu().numpy()

        idx_answer = np.asarray([[1, 1], [0, 1], [1, 1], [1, 1]])
        dist2_answer = np.asarray([[-1, -1], [0.01, -1], [-1, -1], [-1, -1]]).astype(np.float32)

        npt.assert_array_almost_equal(idx, idx_answer)
        npt.assert_array_almost_equal(dist2, dist2_answer)

    def test_random_cpu(self):
        a = torch.randn(1000, 3).to(torch.float)
        b = torch.randn(1500, 3).to(torch.float)
        batch_a = torch.randint(1, (1000,)).sort(0)[0].long()
        batch_b = torch.randint(1, (1500,)).sort(0)[0].long()
        idx, dist2 = ball_query(1.0, 12, a, b, batch_a, batch_b, mode="PARTIAL_DENSE")
        idx, dist2 = ball_query(1.0, 12, b, a, batch_b, batch_a, mode="PARTIAL_DENSE")
        idx = idx.detach().cpu().numpy()
        dist2 = dist2.detach().cpu().numpy()
        idx2, _ = cpu_ball_query(a, b, 1.0, 12)
        print(idx[5], print(idx2[5]))






if __name__ == "__main__":
    unittest.main()
