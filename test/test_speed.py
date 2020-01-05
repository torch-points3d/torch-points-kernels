import unittest
import torch
from torch.autograd import variable
from torch_points import ball_query, three_interpolate, three_interpolate2
import numpy.testing as npt
import numpy as np
from time import time

class TestSpeed(unittest.TestCase):
    def test_simple(self):
        r"""
        Performs weight linear interpolation on 3 features
        Parameters
        ----------
        features : torch.Tensor
            (B, c, m) Features descriptors to be interpolated from
        idx : torch.Tensor
            (B, n, 3) three nearest neighbors of the target features in features
        weight : torch.Tensor
            (B, n, 3) weights

        Returns
        -------
        torch.Tensor
            (B, c, n) tensor of the interpolated features
        """       
        time_history_1 = []
        time_history_2 = []
        for i in range(100):
            batch_size = np.random.randint(2, 8)
            c = np.random.randint(128, 256)
            m = np.random.randint(256, 512)
            n = np.random.randint(2 * m, 4 * m)

            features = np.random.normal(0, 1, (batch_size, c, m))
            idx = np.random.randint(0, m, (batch_size, n, 3))
            weight = np.abs(np.random.normal(0, 1, (batch_size, n, 3)))

            features = torch.tensor(features, requires_grad=True).float().cuda()
            idx = torch.from_numpy(idx).int().cuda()
            weight = torch.from_numpy(weight).float().cuda()

            t0 = time()
            out = three_interpolate(features, idx, weight)
            out.backward(torch.zeros_like(out))
            
            t1 = time()
            
            out2 = three_interpolate2(features, idx, weight)
            out2.backward(torch.zeros_like(out2))
            t2 = time()

            if i > 5:
                time_history_1.append(t1 - t0)
                time_history_2.append(t2 - t1)

        print(np.mean(time_history_1), np.std(time_history_1), np.sum(time_history_1))
        print(np.mean(time_history_2), np.std(time_history_2), np.sum(time_history_2))

if __name__ == "__main__":
    unittest.main()
