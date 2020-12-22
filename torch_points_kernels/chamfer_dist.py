import torch

if torch.cuda.is_available():
    import torch_points_kernels.points_cuda as tpcuda


class ChamferFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        if not torch.cuda.is_available():
            raise NotImplementedError("CPU version is not available for Chamfer Distance")

        dist1, dist2, idx1, idx2 = tpcuda.chamfer_dist(xyz1, xyz2)
        ctx.save_for_backward(xyz1, xyz2, idx1, idx2)

        return dist1, dist2

    @staticmethod
    def backward(ctx, grad_dist1, grad_dist2):
        xyz1, xyz2, idx1, idx2 = ctx.saved_tensors
        grad_xyz1, grad_xyz2 = tpcuda.chamfer_dist_grad(xyz1, xyz2, idx1, idx2, grad_dist1, grad_dist2)
        return grad_xyz1, grad_xyz2


def chamfer_dist(xyz1, xyz2, ignore_zeros=False):
    r"""
    Calcuates the distance between B pairs of point clouds

    Parameters
    ----------
    xyz1 : torch.Tensor (dtype=torch.float32)
        (B, n1, 3) B point clouds containing n1 points
    xyz2 : torch.Tensor (dtype=torch.float32)
        (B, n2, 3) B point clouds containing n2 points
    ignore_zeros : bool
        ignore the point whose coordinate is (0, 0, 0) or not

    Returns
    -------
    dist: torch.Tensor
        (B, ): the distances between B pairs of point clouds
    """
    if len(xyz1.shape) != 3 or xyz1.size(2) != 3 or len(xyz2.shape) != 3 or xyz2.size(2) != 3:
        raise ValueError("The input point cloud should be of size (B, n_pts, 3)")

    batch_size = xyz1.size(0)
    if batch_size == 1 and ignore_zeros:
        non_zeros1 = torch.sum(xyz1, dim=2).ne(0)
        non_zeros2 = torch.sum(xyz2, dim=2).ne(0)
        xyz1 = xyz1[non_zeros1].unsqueeze(dim=0)
        xyz2 = xyz2[non_zeros2].unsqueeze(dim=0)

    dist1, dist2 = ChamferFunction.apply(xyz1, xyz2)
    return torch.mean(dist1) + torch.mean(dist2)
