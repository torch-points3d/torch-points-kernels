import numpy as np
import os
import sys
import torch
import unittest

from torch.autograd import gradcheck

from . import run_if_cuda


ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.insert(0, ROOT)

from torch_points_kernels import ChamferFunction, chamfer_dist


class TestChamferDistance(unittest.TestCase):
    @run_if_cuda
    def test_chamfer_dist_grad(self):
        x = torch.rand(4, 64, 3).double()
        y = torch.rand(4, 128, 3).double()
        x.requires_grad = True
        y.requires_grad = True
        test = gradcheck(ChamferFunction.apply, [x.cuda(), y.cuda()])

    @run_if_cuda
    def test_chamfer_dist(self):
        xyz1 = torch.from_numpy(np.array([[[0, 0, 0], [1, 1, 1], [2, 0, 1]]])).float()
        xyz2 = torch.from_numpy(np.array([[[1, 0, 0], [1, 2, 1]]])).float()
        dist = chamfer_dist(xyz1.cuda(), xyz2.cuda())
        self.assertAlmostEqual(dist.item(), 2.333333, places=5)

    @run_if_cuda
    def test_chamfer_dist_ignore_zeros(self):
        xyz1 = torch.from_numpy(np.array([[[0, 0, 0], [1, 1, 1], [2, 0, 1]]])).float()
        xyz2 = torch.from_numpy(np.array([[[1, 0, 0], [1, 2, 1]]])).float()
        dist = chamfer_dist(xyz1.cuda(), xyz2.cuda(), True)
        self.assertAlmostEqual(dist.item(), 3.0, places=5)


if __name__ == "__main__":
    unittest.main()
