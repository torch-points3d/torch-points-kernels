import unittest
import torch
import os
import sys
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.insert(0, ROOT)

from torch_points_kernels.metrics import instance_iou


class TestInstanceIou(unittest.TestCase):
    def test_simple(self):
        gt_instances = torch.tensor([1, 2, 1, 2, 2, 3, 0])
        proposed_instances = [
            torch.tensor([0, 2]),  # 100% instance 1
            torch.tensor([1, 4]),  # 2/3 of instance 2
            torch.tensor([3, 5]),  # 1/3 of instance 2 and 1/1 of instance 3
        ]

        ious = instance_iou(proposed_instances, gt_instances)
        torch.testing.assert_allclose(
            ious, torch.tensor([[1, 0, 0], [0, 2 / 3.0, 0], [0, 1.0 / 4.0, 1.0 / 2.0]])
        )


if __name__ == "__main__":
    unittest.main()
