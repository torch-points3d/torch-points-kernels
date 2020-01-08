import unittest
import torch
from torch_points import ball_query, ball_query_shifted
import numpy.testing as npt
import numpy as np


class TestBall(unittest.TestCase):
    def test_simple(self):
        a = torch.tensor([[[0, 0, 0], [1, 0, 0], [2, 0, 0]]]).to(torch.float).cuda()
        b = torch.tensor([[[0, 0, 0]]]).to(torch.float).cuda()

        npt.assert_array_equal(ball_query(1, 2, a, b).detach().cpu().numpy(), np.array([[[0, 0]]]))

class TestBallShifted(unittest.TestCase):
    def test_wi_shadow(self):
        
        a = torch.tensor([[[0, 0, 0], [1, 0, 0], [2, 0, 0]]]).to(torch.float).cuda()
        b = torch.tensor([[[0, 0, 0]]]).to(torch.float).cuda()

        npt.assert_array_equal(ball_query_shifted(1, 2, a, b).detach().cpu().numpy(), np.array([[[1, 0]]]))

    def test_wo_shadow(self):
        
        a = torch.tensor([[[0, 0, 0], [0.5, 0, 0], [2, 0, 0]]]).to(torch.float).cuda()
        b = torch.tensor([[[0, 0, 0]]]).to(torch.float).cuda()

        npt.assert_array_equal(ball_query_shifted(1, 2, a, b).detach().cpu().numpy(), np.array([[[1, 2]]]))


if __name__ == "__main__":
    unittest.main()
