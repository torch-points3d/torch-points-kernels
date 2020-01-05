import torch
from torch.autograd import Function
import torch.nn as nn
import sys

import torch_points.points_cuda as tpcuda


class FurthestPointSampling(Function):
    @staticmethod
    def forward(ctx, xyz, npoint):
        return tpcuda.furthest_point_sampling(xyz, npoint)

    @staticmethod
    def backward(xyz, a=None):
        return None, None


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
    return FurthestPointSampling.apply(xyz, npoint)


class GatherOperation(Function):
    @staticmethod
    def forward(ctx, features, idx):
        # type: (Any, torch.Tensor, torch.Tensor) -> torch.Tensor
        _, C, N = features.size()

        ctx.for_backwards = (idx, C, N)

        return tpcuda.gather_points(features, idx)

    @staticmethod
    def backward(ctx, grad_out):
        idx, C, N = ctx.for_backwards

        grad_features = tpcuda.gather_points_grad(grad_out.contiguous(), idx, N)
        return grad_features, None


def gather_operation(features, idx):
    r"""

       Parameters
       ----------
       features : torch.Tensor
           (B, C, N) tensor

       idx : torch.Tensor
           (B, npoint) tensor of the features to gather

       Returns
       -------
       torch.Tensor
           (B, C, npoint) tensor
       """
    return GatherOperation.apply(features, idx)


class ThreeNN(Function):
    @staticmethod
    def forward(ctx, unknown, known):
        # type: (Any, torch.Tensor, torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
        dist2, idx = tpcuda.three_nn(unknown, known)

        return torch.sqrt(dist2), idx

    @staticmethod
    def backward(ctx, a=None, b=None):
        return None, None


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
    return ThreeNN.apply(unknown, known)


class ThreeInterpolate(Function):
    @staticmethod
    def forward(ctx, features, idx, weight):
        # type(Any, torch.Tensor, torch.Tensor, torch.Tensor) -> Torch.Tensor
        B, c, m = features.size()
        n = idx.size(1)

        ctx.three_interpolate_for_backward = (idx, weight, m)

        return tpcuda.three_interpolate(features, idx, weight)

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

        grad_features = tpcuda.three_interpolate_grad(
            grad_out.contiguous(), idx, weight, m
        )

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

class ThreeInterpolate2(Function):
    @staticmethod
    def forward(ctx, features, idx, weight):
        # type(Any, torch.Tensor, torch.Tensor, torch.Tensor) -> Torch.Tensor
        B, c, m = features.size()
        n = idx.size(1)

        ctx.three_interpolate_for_backward = (idx, weight, m)

        out = torch.zeros((B, c, n)).to(features.device)
        tpcuda.three_interpolate_wrapper_fast(B, c, m, n, features, idx, weight, out)
        return out

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

        B, c, n = grad_out.size()

        grad_features = torch.zeros(B, c, m).to(grad_out.device)
        grad_out_data = grad_out.data.contiguous()

        tpcuda.three_interpolate_grad_wrapper_fast(B, c, n, m, grad_out_data, idx, weight, grad_features.data)
        return grad_features, None, None

def three_interpolate2(features, idx, weight):
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
    return ThreeInterpolate2.apply(features, idx, weight)


class GroupingOperation(Function):
    @staticmethod
    def forward(ctx, features, idx):
        # type: (Any, torch.Tensor, torch.Tensor) -> torch.Tensor
        B, nfeatures, nsample = idx.size()
        _, C, N = features.size()

        ctx.for_backwards = (idx, N)

        return tpcuda.group_points(features, idx)

    @staticmethod
    def backward(ctx, grad_out):
        # type: (Any, torch.tensor) -> Tuple[torch.Tensor, torch.Tensor]
        r"""

        Parameters
        ----------
        grad_out : torch.Tensor
            (B, C, npoint, nsample) tensor of the gradients of the output from forward

        Returns
        -------
        torch.Tensor
            (B, C, N) gradient of the features
        None
        """
        idx, N = ctx.for_backwards

        grad_features = tpcuda.group_points_grad(grad_out.contiguous(), idx, N)

        return grad_features, None


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
    return GroupingOperation.apply(features, idx)


class BallQuery(Function):
    @staticmethod
    def forward(ctx, radius, nsample, xyz, new_xyz):
        # type: (Any, float, int, torch.Tensor, torch.Tensor) -> torch.Tensor
        return tpcuda.ball_query(new_xyz, xyz, radius, nsample)

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None


def ball_query(radius, nsample, xyz, new_xyz):
    r"""
    Parameters
    ----------
    radius : float
        radius of the balls
    nsample : int
        maximum number of features in the balls
    xyz : torch.Tensor
        (B, N, 3) xyz coordinates of the features
    new_xyz : torch.Tensor
        (B, npoint, 3) centers of the ball query

    Returns
    -------
    torch.Tensor
        (B, npoint, nsample) tensor with the indicies of the features that form the query balls
    """
    return BallQuery.apply(radius, nsample, xyz, new_xyz)
