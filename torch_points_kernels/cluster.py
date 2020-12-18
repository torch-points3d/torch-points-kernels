import torch
from .torchpoints import ball_query_partial_dense
import numpy as np
import numba
from typing import List


@numba.jit(nopython=True)
def _grow_proximity_core(neighbours, min_cluster_size):
    num_points = int(neighbours.shape[0])
    visited = np.zeros((num_points,), dtype=numba.types.bool_)
    clusters = []
    for i in range(num_points):
        if visited[i]:
            continue

        cluster = []
        queue = []
        visited[i] = True
        queue.append(i)
        cluster.append(i)
        while len(queue):
            k = queue.pop()
            k_neighbours = neighbours[k]
            for nei in k_neighbours:
                if nei.item() == -1:
                    break

                if not visited[nei]:
                    visited[nei] = True
                    queue.append(nei.item())
                    cluster.append(nei.item())

        if len(cluster) >= min_cluster_size:
            clusters.append(cluster)

    return clusters


def grow_proximity(pos, batch, nsample=16, radius=0.02, min_cluster_size=32):
    """Grow based on proximity only
    Neighbour search is done on device while the cluster assignement is done on cpu"""
    assert pos.shape[0] == batch.shape[0]
    neighbours = ball_query_partial_dense(radius, nsample, pos, pos, batch, batch)[0].cpu().numpy()
    return _grow_proximity_core(neighbours, min_cluster_size)


def region_grow(
    pos, labels, batch, ignore_labels=[], nsample=16, radius=0.02, min_cluster_size=32
) -> List[torch.Tensor]:
    """Region growing clustering algorithm proposed in
    PointGroup: Dual-Set Point Grouping for 3D Instance Segmentation
    https://arxiv.org/pdf/2004.01658.pdf
    for instance segmentation

    Parameters
    ----------
    pos: torch.Tensor [N, 3]
        Location of the points
    labels: torch.Tensor [N,]
        labels of each point
    ignore_labels:
        Labels that should be ignored, no region growing will be performed on those
    nsample:
        maximum number of neighbours to consider
    radius:
        radius for the neighbour search
    min_cluster_size:
        Number of points above which a cluster is considered valid
    """
    assert labels.dim() == 1
    assert pos.dim() == 2
    assert pos.shape[0] == labels.shape[0]

    unique_labels = torch.unique(labels)
    clusters = []
    ind = torch.arange(0, pos.shape[0])
    for l in unique_labels:
        if l in ignore_labels:
            continue

        # Build clusters for a given label (ignore other points)
        label_mask = labels == l
        local_ind = ind[label_mask]

        # Remap batch to a continuous sequence
        label_batch = batch[label_mask]
        unique_in_batch = torch.unique(label_batch)
        remaped_batch = torch.empty_like(label_batch)
        for new, old in enumerate(unique_in_batch):
            mask = label_batch == old
            remaped_batch[mask] = new

        # Cluster
        label_clusters = grow_proximity(
            pos[label_mask, :],
            remaped_batch,
            nsample=nsample,
            radius=radius,
            min_cluster_size=min_cluster_size,
        )

        # Remap indices to original coordinates
        if len(label_clusters):
            for cluster in label_clusters:
                cluster = torch.tensor(cluster).to(pos.device)
                clusters.append(local_ind[cluster])

    return clusters
