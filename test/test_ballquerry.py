import unittest
import torch
from torch_points import ball_query
import numpy.testing as npt
import numpy as np


class TestBall(unittest.TestCase):
    def test_simple(self):
        a = torch.tensor([[[0, 0, 0], [1, 0, 0], [2, 0, 0]]]).to(torch.float).cuda()
        b = torch.tensor([[[0, 0, 0]]]).to(torch.float).cuda()

        npt.assert_array_equal(ball_query(1, 2, a, b).detach().cpu().numpy(), np.array([[[0, 0]]]))


if __name__ == "__main__":
    unittest.main()
