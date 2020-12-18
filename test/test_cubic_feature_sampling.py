import numpy as np
import os
import sys
import torch
import unittest

from torch.autograd import gradcheck

from . import run_if_cuda


ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.insert(0, ROOT)

from torch_points_kernels.cubic_feature_sampling import CubicFeatureSamplingFunction, cubic_feature_sampling


class TestCubicFeatureSampling(unittest.TestCase):
    @run_if_cuda
    def test_neighborhood_size_1(self):
        ptcloud = torch.rand(2, 64, 3) * 2 - 1
        cubic_features = torch.rand(2, 4, 8, 8, 8)
        ptcloud.requires_grad = True
        cubic_features.requires_grad = True
        self.assertTrue(
            gradcheck(
                CubicFeatureSamplingFunction.apply,
                [ptcloud.double().cuda(), cubic_features.double().cuda()],
            )
        )

    @run_if_cuda
    def test_neighborhood_size_2(self):
        ptcloud = torch.rand(2, 32, 3) * 2 - 1
        cubic_features = torch.rand(2, 2, 8, 8, 8)
        ptcloud.requires_grad = True
        cubic_features.requires_grad = True
        self.assertTrue(
            gradcheck(
                CubicFeatureSamplingFunction.apply,
                [ptcloud.double().cuda(), cubic_features.double().cuda(), 2],
            )
        )

    @run_if_cuda
    def test_neighborhood_size_3(self):
        ptcloud = torch.rand(1, 32, 3) * 2 - 1
        cubic_features = torch.rand(1, 2, 16, 16, 16)
        ptcloud.requires_grad = True
        cubic_features.requires_grad = True
        self.assertTrue(
            gradcheck(
                CubicFeatureSamplingFunction.apply,
                [ptcloud.double().cuda(), cubic_features.double().cuda(), 3],
            )
        )


if __name__ == "__main__":
    unittest.main()
