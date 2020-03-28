import unittest
import torch
from torch.autograd import gradcheck
from torch_points_kernels import three_interpolate, three_nn

from . import run_if_cuda


class TestInterpolate(unittest.TestCase):
    @run_if_cuda
    def test_gpu(self):
        pos = torch.randn([16, 100, 3]).cuda()
        pos_skip = torch.randn([16, 500, 3]).cuda()
        x = torch.randn([16, 30, 100], requires_grad=True).cuda()

        dist, idx = three_nn(pos_skip, pos)
        dist_recip = 1.0 / (dist + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm
        interpolated_feats = three_interpolate(x, idx, weight)

        dist, idx = three_nn(pos_skip.cpu(), pos.cpu())
        dist_recip = 1.0 / (dist + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm
        interpolated_feats_cpu = three_interpolate(x.cpu(), idx, weight)

        torch.testing.assert_allclose(interpolated_feats_cpu, interpolated_feats.cpu())

    def test_grad(self):
        b, n, k = (2, 10, 3)
        pos = torch.randn([b, n, k]).double()
        pos_skip = torch.randn([b, 2 * n, k]).double()
        x = torch.randn([b, 30, n], requires_grad=True).double()
        dist, idx = three_nn(pos_skip, pos)
        dist_recip = 1.0 / (dist + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm
        input = (x, idx, weight.double())
        test = gradcheck(three_interpolate, input, eps=1e-6, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
