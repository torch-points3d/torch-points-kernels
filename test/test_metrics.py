import unittest
import torch
import os
import sys
import numpy as np
import random

ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.insert(0, ROOT)

from torch_points_kernels.metrics import instance_iou
from test import run_if_cuda


class TestInstanceIou(unittest.TestCase):
    def test_simple(self):
        gt_instances = torch.tensor([1, 2, 1, 2, 2, 3, 0])
        proposed_instances = [
            torch.tensor([0, 2]),  # 100% instance 1
            torch.tensor([1, 4]),  # 2/3 of instance 2
            torch.tensor([3, 5]),  # 1/3 of instance 2 and 1/1 of instance 3
        ]

        ious = instance_iou(proposed_instances, gt_instances)
        torch.testing.assert_allclose(ious, torch.tensor([[1, 0, 0], [0, 2 / 3.0, 0], [0, 1.0 / 4.0, 1.0 / 2.0]]))

    @run_if_cuda
    def test_simple_cuda(self):
        gt_instances = torch.tensor([1, 2, 1, 2, 2, 3, 0]).cuda()
        proposed_instances = [
            torch.tensor([0, 2]).cuda(),  # 100% instance 1
            torch.tensor([1, 4]).cuda(),  # 2/3 of instance 2
            torch.tensor([3, 5]).cuda(),  # 1/3 of instance 2 and 1/1 of instance 3
        ]

        ious = instance_iou(proposed_instances, gt_instances)
        torch.testing.assert_allclose(
            ious, torch.tensor([[1, 0, 0], [0, 2 / 3.0, 0], [0, 1.0 / 4.0, 1.0 / 2.0]]).cuda(),
        )

    @run_if_cuda
    def test_same(self):
        gt_instances = torch.randint(0, 10, (1000,))
        proposed_instances = []
        for i in range(20):
            instance_size = random.randint(5, 50)
            proposed_instances.append(torch.randint(0, 1000, (instance_size,)))

        ious = instance_iou(proposed_instances, gt_instances)
        proposed_instances_cuda = [i.cuda() for i in proposed_instances]
        ious_cuda = instance_iou(proposed_instances_cuda, gt_instances.cuda())

        torch.testing.assert_allclose(ious, ious_cuda.cpu())


if __name__ == "__main__":
    unittest.main()
