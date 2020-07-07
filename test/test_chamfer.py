import os
import sys
import torch
import unittest

from torch.autograd import gradcheck

ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.insert(0, ROOT)

from torch_points_kernels import ChamferFunction


class TestChamferDistance(unittest.TestCase):
    def test_chamfer_dist(self):
        x = torch.rand(4, 64, 3).double()
        y = torch.rand(4, 128, 3).double()
        x.requires_grad = True
        y.requires_grad = True
        test = gradcheck(ChamferFunction.apply, [x.cuda(), y.cuda()])


if __name__ == '__main__':
    unittest.main()
