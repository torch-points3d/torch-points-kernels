import torch

if torch.cuda.is_available():
    import torch_points_kernels.points_cuda as tpcuda


class CubicFeatureSamplingFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ptcloud, cubic_features, neighborhood_size=1):
        scale = cubic_features.size(2)
        if not torch.cuda.is_available():
            raise NotImplementedError("CPU version is not available for Cubic Feature Sampling")

        point_features, grid_pt_indexes = tpcuda.cubic_feature_sampling(
            scale, neighborhood_size, ptcloud, cubic_features
        )
        ctx.save_for_backward(torch.Tensor([scale]), torch.Tensor([neighborhood_size]), grid_pt_indexes)
        return point_features

    @staticmethod
    def backward(ctx, grad_point_features):
        scale, neighborhood_size, grid_pt_indexes = ctx.saved_tensors
        scale = int(scale.item())
        neighborhood_size = int(neighborhood_size.item())
        grad_point_features = grad_point_features.contiguous()
        grad_ptcloud, grad_cubic_features = tpcuda.cubic_feature_sampling_grad(
            scale, neighborhood_size, grad_point_features, grid_pt_indexes
        )
        return grad_ptcloud, grad_cubic_features, None


def cubic_feature_sampling(ptcloud, cubic_features, neighborhood_size=1):
    r"""
    Sample the features of points from 3D feature maps that the point lies in.
    Please refer to https://arxiv.org/pdf/2006.03761 for more information

    Parameters
    ----------
    ptcloud : torch.Tensor (dtype=torch.float32)
        (B, n_pts, 3) point clouds containing n_pts points
    cubic_features : torch.Tensor (dtype=torch.float32)
        (B, c, m, m, m) 3D feature maps of sizes m x m x m and c channels
    neighborhood_size : int
        The neighborhood cubes to sample.
        neighborhood_size = 1 means to sample the cube that point lies in.
        neighborhood_size = 2 means to sample surrouding cubes (step = 1) of
        the cube that point lies in.

    Returns
    -------
    dist: torch.Tensor
        (B, n_pts, n_vertices, c), where n_vertices = (neighborhood_size * 2)^3
    """
    if len(ptcloud.shape) != 3 or ptcloud.shape[2] != 3:
        raise ValueError("The input point cloud should be of size (B, n_pts, 3).")

    h_scale = cubic_features.size(2) / 2
    ptcloud = ptcloud * h_scale + h_scale
    return CubicFeatureSamplingFunction.apply(ptcloud, cubic_features, neighborhood_size)
