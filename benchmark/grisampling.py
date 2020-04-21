import torch
import time
import os
import sys
import re
import torch.nn.functional as F
from torch_geometric.nn import voxel_grid
from torch_scatter import scatter_mean, scatter_add
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_geometric.data import Data
from torch_cluster import grid_cluster

ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.insert(0, ROOT)

from torch_points_kernels import GridSampler

def group_data(data, cluster=None, unique_pos_indices=None, mode="last", skip_keys=[]):
    """ Group data based on indices in cluster. 
    The option ``mode`` controls how data gets agregated within each cluster.
    
    Parameters
    ----------
    data : Data
        [description]
    cluster : torch.Tensor
        Tensor of the same size as the number of points in data. Each element is the cluster index of that point.
    unique_pos_indices : torch.tensor
        Tensor containing one index per cluster, this index will be used to select features and labels
    mode : str
        Option to select how the features and labels for each voxel is computed. Can be ``last`` or ``mean``.
        ``last`` selects the last point falling in a voxel as the representent, ``mean`` takes the average.
    skip_keys: list
        Keys of attributes to skip in the grouping
    """

    assert mode in ["mean", "last"]
    if mode == "mean" and cluster is None:
        raise ValueError("In mean mode the cluster argument needs to be specified")
    if mode == "last" and unique_pos_indices is None:
        raise ValueError("In last mode the unique_pos_indices argument needs to be specified")

    num_nodes = data.num_nodes
    for key, item in data:
        if bool(re.search("edge", key)):
            raise ValueError("Edges not supported. Wrong data type.")
        if key in skip_keys:
            continue

        if torch.is_tensor(item) and item.size(0) == num_nodes:
            if mode == "last" or key == "batch":
                data[key] = item[unique_pos_indices]
            elif mode == "mean":
                if key == "y":
                    item_min = item.min()
                    item = F.one_hot(item - item_min)
                    item = scatter_add(item, cluster, dim=0)
                    data[key] = item.argmax(dim=-1) + item_min
                else:
                    data[key] = scatter_mean(item, cluster, dim=0)
    return data


class GridSampling:
    """ Clusters points into voxels with size :attr:`size`.

    Parameters
    ----------
    size: float
        Size of a voxel (in each dimension).
    quantize_coords: bool
            If True, it will convert the points into their associated sparse coordinates within the grid. \
    mode: string:
        The mode can be either `last` or `mean`.
        If mode is `mean`, all the points and their features within a cell will be averaged
        If mode is `last`, one random points per cell will be selected with its associated features
    """

    def __init__(self, size, quantize_coords=False, mode="mean", verbose=False):
        self._grid_size = size
        self._quantize_coords = quantize_coords
        self._mode = mode
        if verbose:
            log.warning(
                "If you need to keep track of the position of your points, use SaveOriginalPosId transform before using GridSampling"
            )

            if self._mode == "last":
                log.warning(
                    "The tensors within data will be shuffled each time this transform is applied. Be careful that if an attribute doesn't have the size of num_points, it won't be shuffled"
                )

    def _process(self, data):
        coords = ((data.pos) / self._grid_size).int()
        if "batch" not in data:
            cluster = grid_cluster(coords, torch.tensor([1, 1, 1]))
        else:
            cluster = voxel_grid(coords, data.batch, 1)
        cluster, unique_pos_indices = consecutive_cluster(cluster)

        skip_keys = []
        if self._quantize_coords:
            data.pos = coords[unique_pos_indices]
            skip_keys.append("pos")

        data = group_data(data, cluster, unique_pos_indices, mode=self._mode, skip_keys=skip_keys)
        return data

    def __call__(self, data):
        if isinstance(data, list):
            data = [self._process(d) for d in data]
        else:
            data = self._process(data)
        return data

    def __repr__(self):
        return "{}(grid_size={}, quantize_coords={}, mode={})".format(
            self.__class__.__name__, self._grid_size, self._quantize_coords, self._mode
        )

if __name__ == "__main__":
    num_points = 100000
    pos = torch.rand((num_points,3)) * 5
    feat = torch.rand((num_points,5))
    y = torch.randint(0,10,(num_points,))

    data = Data(pos=pos, feat = feat, y = y)

    runs = 100
    for grid_size in [.1, .5, 1]:
        current_method_times = []
        gr = GridSampling(grid_size)
        for i in range(runs):
            start = time.time()
            d = data.clone()
            d = gr(d)
            current_method_times.append(time.time() - start)
        print("Grid size %.2f, legacy: %.3fs" % (grid_size, sum(current_method_times) / runs))

        current_method_times = []
        fit_time = []
        max_time = []
        for i in range(runs):
            start = time.time()
            gr = GridSampler(grid_size)
            gr.fit(pos)
            fit_time.append(time.time() - start)
            gr.aggregate(pos)
            gr.aggregate(feat)
            start_count = time.time()
            gr.aggregate(y, mode="max_count")
            max_time.append(time.time() - start_count)
            current_method_times.append(time.time() - start)
        print("Grid size %.2f, New: %.3fs, fit time: %.3fs, max_cout: %.3f" % (grid_size, sum(current_method_times) / runs, sum(fit_time) / runs, sum(max_time)/ runs))


    