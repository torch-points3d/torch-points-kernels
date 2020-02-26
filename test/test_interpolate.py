import unittest
import torch
from torch_points import three_interpolate_tg, three_interpolate, three_nn

class TestInterpolate(unittest.TestCase):
    def test_cpu(self):
        pos = torch.randn([16, 100, 3])
        pos_skip = torch.randn([16, 500, 3])
        x = torch.randn([16, 30, 100])

        # # dense
        # dist, idx = three_nn(pos_skip, pos)
        # dist_recip = 1.0 / (dist + 1e-8)
        # norm = torch.sum(dist_recip, dim=2, keepdim=True)
        # weight = dist_recip / norm
        # interpolated_feats = three_interpolate(x, idx, weight)

        # sparse
        sp_interpolated = three_interpolate_tg(x,pos,pos_skip)

        # torch.testing.assert_allclose(sp_interpolated, interpolated_feats)

if __name__ == "__main__":
    unittest.main()