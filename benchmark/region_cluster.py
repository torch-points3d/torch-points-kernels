import unittest
import torch
import os
import sys
import time
import random

ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.insert(0, ROOT)

from torch_points_kernels.cluster import grow_proximity

torch.manual_seed(0)

num_points = 100000
pos1 = torch.rand((num_points, 3))
pos2 = torch.rand((num_points, 3)) + 2
pos3 = torch.rand((num_points, 3)) + 4
labels1 = torch.ones(num_points).long()
labels2 = torch.ones(num_points).long()
labels3 = torch.ones(num_points).long()
pos = torch.cat([pos1, pos2, pos3], 0)
label = torch.cat([labels1, labels2, labels3], 0)
batch = torch.ones((3 * num_points)).long()
cl = grow_proximity(pos, batch, radius=0.5)


import cProfile, pstats, io
from pstats import SortKey

pr = cProfile.Profile()
pr.enable()
t_start = time.perf_counter()
grow_proximity(pos, batch)
print(time.perf_counter() - t_start)
pr.disable()
s = io.StringIO()
sortby = SortKey.CUMULATIVE
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()
print(s.getvalue())
