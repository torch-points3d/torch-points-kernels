import unittest
import torch
from torch_points import ball_query
from torch_cluster import radius_cuda
import numpy.testing as npt
import numpy as np


class TestBallPartial(unittest.TestCase):
    def test_simple_gpu(self):
        x = torch.tensor([[10, 0, 0], [0.1, 0, 0], [10, 0, 0], [0.1, 0, 0]]).to(torch.float).cuda()
        y = torch.tensor([[0, 0, 0]]).to(torch.float).cuda()
        batch_x = torch.from_numpy(np.asarray([0, 0, 1, 1])).long().cuda()
        batch_y = torch.from_numpy(np.asarray([0])).long().cuda()
        
        print(x.shape, y.shape, batch_x.shape, batch_y.shape)

        batch_x = torch.from_numpy(np.asarray([0, 0, 1, 1])).long().cuda()
        batch_y = torch.from_numpy(np.asarray([0])).long().cuda()

        idx, dist2 = ball_query(1, 2, x, y, batch_x, batch_y, mode="PARTIAL_DENSE")

        idx = idx.detach().cpu().numpy()
        dist2 = dist2.detach().cpu().numpy()

        print(idx)
        print(dist2)

        #npt.assert_array_equal(idx, )
        #npt.assert_array_equal(dist2, np.asarray([[0.3, 0.1]]))



if __name__ == "__main__":
    unittest.main()
