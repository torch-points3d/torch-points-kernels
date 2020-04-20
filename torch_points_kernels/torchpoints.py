import torch
from torch.autograd import Function
import torch.nn as nn
import sys
from typing import Optional, Any, Tuple

import torch_points_kernels.points_cpu as tpcpu
from .knn import knn

if torch.cuda.is_available():
    import torch_points_kernels.points_cuda as tpcuda


def furthest_point_sample(xyz, npoint):
    # type: (Any, torch.Tensor, int) -> torch.Tensor
    r"""
    Uses iterative furthest point sampling to select a set of npoint features that have the largest
    minimum distance

    Parameters
    ----------
    xyz : torch.Tensor
        (B, N, 3) tensor where N > npoint
    npoint : int32
        number of features in the sampled set

    Returns
    -------
    torch.Tensor
        (B, npoint) tensor containing the set
    """
    if npoint > xyz.shape[1]:
        raise ValueError("caanot sample %i points from an input set of %i points" % (npoint, xyz.shape[1]))
    if xyz.is_cuda:
        return tpcuda.furthest_point_sampling(xyz, npoint)
    else:
        return tpcpu.fps(xyz, npoint, True)


def three_nn(unknown, known):
    r"""
        Find the three nearest neighbors of unknown in known
    Parameters
    ----------
    unknown : torch.Tensor
        (B, n, 3) tensor of known features
    known : torch.Tensor
        (B, m, 3) tensor of unknown features

    Returns
    -------
    dist : torch.Tensor
        (B, n, 3) l2 distance to the three nearest neighbors
    idx : torch.Tensor
        (B, n, 3) index of 3 nearest neighbors
    """
    if unknown.shape[1] < 3:
        raise ValueError("Not enough points. unknown should ahve at least 3 points.")
    if unknown.is_cuda:
        dist2, idx = tpcuda.three_nn(unknown, known)
    else:
        idx, dist2 = knn(known, unknown, 3)

    return torch.sqrt(dist2), idx


class ThreeInterpolate(Function):
    @staticmethod
    def forward(ctx, features, idx, weight):
        # type(Any, torch.Tensor, torch.Tensor, torch.Tensor) -> Torch.Tensor
        B, c, m = features.size()
        n = idx.size(1)

        ctx.three_interpolate_for_backward = (idx, weight, m)

        if features.is_cuda:
            return tpcuda.three_interpolate(features, idx, weight)
        else:
            return tpcpu.knn_interpolate(features, idx, weight)

    @staticmethod
    def backward(ctx, grad_out):
        # type: (Any, torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        r"""
        Parameters
        ----------
        grad_out : torch.Tensor
            (B, c, n) tensor with gradients of ouputs

        Returns
        -------
        grad_features : torch.Tensor
            (B, c, m) tensor with gradients of features

        None

        None
        """
        idx, weight, m = ctx.three_interpolate_for_backward

        if grad_out.is_cuda:
            grad_features = tpcuda.three_interpolate_grad(grad_out.contiguous(), idx, weight, m)
        else:
            grad_features = tpcpu.knn_interpolate_grad(grad_out.contiguous(), idx, weight, m)

        return grad_features, None, None


def three_interpolate(features, idx, weight):
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
    return ThreeInterpolate.apply(features, idx, weight)


def grouping_operation(features, idx):
    r"""
    Parameters
    ----------
    features : torch.Tensor
        (B, C, N) tensor of features to group
    idx : torch.Tensor
        (B, npoint, nsample) tensor containing the indicies of features to group with

    Returns
    -------
    torch.Tensor
        (B, C, npoint, nsample) tensor
    """
    all_idx = idx.reshape(idx.shape[0], -1)
    all_idx = all_idx.unsqueeze(1).repeat(1, features.shape[1], 1)
    grouped_features = features.gather(2, all_idx)
    return grouped_features.reshape(idx.shape[0], features.shape[1], idx.shape[1], idx.shape[2])


def ball_query_dense(radius, nsample, xyz, new_xyz, batch_xyz=None, batch_new_xyz=None, sort=False):
    # type: (Any, float, int, torch.Tensor, torch.Tensor) -> torch.Tensor
    if new_xyz.is_cuda:
        if sort:
            raise NotImplementedError("CUDA version does not sort the neighbors")
        ind, dist = tpcuda.ball_query_dense(new_xyz, xyz, radius, nsample)
    else:
        ind, dist = tpcpu.dense_ball_query(new_xyz, xyz, radius, nsample, mode=0, sorted=sort)
    return ind, dist


def ball_query_partial_dense(radius, nsample, x, y, batch_x, batch_y, sort=False):
    # type: (Any, float, int, torch.Tensor, torch.Tensor) -> torch.Tensor
    if x.is_cuda:
        if sort:
            raise NotImplementedError("CUDA version does not sort the neighbors")
        ind, dist = tpcuda.ball_query_partial_dense(x, y, batch_x, batch_y, radius, nsample)
    else:
        ind, dist = tpcpu.batch_ball_query(x, y, batch_x, batch_y, radius, nsample, mode=0, sorted=sort)
    return ind, dist


def ball_query(
    radius: float,
    nsample: int,
    x: torch.Tensor,
    y: torch.Tensor,
    mode: Optional[str] = "dense",
    batch_x: Optional[torch.tensor] = None,
    batch_y: Optional[torch.tensor] = None,
    sort: Optional[bool] = False,
) -> torch.Tensor:
    """
    Arguments:
        radius {float} -- radius of the balls
        nsample {int} -- maximum number of features in the balls
        x {torch.Tensor} --
            (M, 3) [partial_dense] or (B, M, 3) [dense] xyz coordinates of the features
        y {torch.Tensor} --
            (npoint, 3) [partial_dense] or or (B, npoint, 3) [dense] centers of the ball query
        mode {str} -- switch between "dense" or "partial_dense" data layout

    Keyword Arguments:
        batch_x -- (M, ) [partial_dense] or (B, M, 3) [dense] Contains indexes to indicate within batch it belongs to.
        batch_y -- (N, ) Contains indexes to indicate within batch it belongs to
        sort -- bool wether the neighboors are sorted or not (closests first)

    Returns:
        idx: (npoint, nsample) or (B, npoint, nsample) [dense] It contains the indexes of the element within x at radius distance to y
        dist: (N, nsample) or (B, npoint, nsample)  Default value: -1.
                 It contains the squared distance of the element within x at radius distance to y
    """
    if mode is None:
        raise Exception('The mode should be defined within ["partial_dense | dense"]')

    if mode.lower() == "partial_dense":
        if (batch_x is None) or (batch_y is None):
            raise Exception("batch_x and batch_y should be provided")
        assert x.size(0) == batch_x.size(0)
        assert y.size(0) == batch_y.size(0)
        assert x.dim() == 2
        return ball_query_partial_dense(radius, nsample, x, y, batch_x, batch_y, sort=sort)

    elif mode.lower() == "dense":
        if (batch_x is not None) or (batch_y is not None):
            raise Exception("batch_x and batch_y should not be provided")
        assert x.dim() == 3
        return ball_query_dense(radius, nsample, x, y, sort=sort)
    else:
        raise Exception("unrecognized mode {}".format(mode))
