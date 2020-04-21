import os
import sys
import torch
import unittest

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(DIR_PATH, ".."))

from torch_points_kernels import GridSampler


class TestGridSampling(unittest.TestCase):
    def test_simple(self):
        """
        This test verifies that the class output is correct and corresponds to the maximun vote from sub_part
        """
        pos = torch.tensor([[0, 0, 0.1], [0.1, 0, 0], [0, 0.1, 0], [0, 0.1, 0], [0, 0.1, 0],[0.1, 0, 0.1]])
        y = torch.tensor([0,0,2,1,1,3]).long()
        sampler = GridSampler(0.1)
        sampler.fit(pos)
        pos_vox = sampler.aggregate(pos, "mean")
        y_vox = sampler.aggregate(y, "first")
        self.assertEqual(y_vox.shape, (4,))
        for t in [0,2,3]:
            self.assertIn(t, y_vox)
        self.assertNotIn(1,y_vox)

        y_vox = sampler.aggregate(y, "max_count")
        self.assertEqual(y_vox.shape, (4,))
        for t in [0,1,3]:
            self.assertIn(t, y_vox)
        self.assertNotIn(2,y_vox)  

    def test_fails(self):
        pos = torch.tensor([[0, 0, 0.1], [0.1, 0, 0], [0, 0.1, 0]])
        sampler = GridSampler(0.1)
        sampler.fit(pos)
        with self.assertRaises(RuntimeError):
            sampler.aggregate(torch.tensor([1]), mode="max_count")
        with self.assertRaises(RuntimeError):
            sampler.aggregate(torch.tensor([1,1,1]).int(), mode="mean")
        with self.assertRaises(RuntimeError):
            sampler.aggregate(torch.tensor([1,1,1]).float(), mode="max_count")

if __name__ == "__main__":
    unittest.main()
