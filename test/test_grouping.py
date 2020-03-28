import unittest
import torch
import numpy as np
import numpy.testing as npt
from torch_points_kernels import grouping_operation


class TestGroup(unittest.TestCase):

    # input: points(b, c, n) idx(b, npoints, nsample)
    # output: out(b, c, npoints, nsample)
    def test_simple(self):
        features = torch.tensor(
            [
                [[0, 10, 0], [1, 11, 0], [2, 12, 0]],
                [[100, 110, 120], [101, 111, 121], [102, 112, 122],],  # x-coordinates  # y-coordinates  # z-coordinates
            ]
        ).type(torch.float)
        idx = torch.tensor([[[1, 0], [0, 0]], [[0, 1], [1, 2]]]).type(torch.long)

        expected = np.array(
            [
                [[[10, 0], [0, 0]], [[11, 1], [1, 1]], [[12, 2], [2, 2]]],
                [  # 2nd batch
                    [  # x-coordinates
                        [100, 110],  # x-coordinates of samples for point 0
                        [110, 120],  # x-coordinates of samples for point 1
                    ],
                    [[101, 111], [111, 121]],  # y-coordinates
                    [[102, 112], [112, 122]],  # z-coordinates
                ],
            ]
        )

        cpu_output = grouping_operation(features, idx).detach().cpu().numpy()

        npt.assert_array_equal(expected, cpu_output)

        if torch.cuda.is_available():
            npt.assert_array_equal(grouping_operation(features.cuda(), idx.cuda()).detach().cpu().numpy(), expected)


if __name__ == "__main__":
    unittest.main()
