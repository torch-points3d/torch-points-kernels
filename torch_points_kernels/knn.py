import torch_points_kernels.points_cpu as tpcpu


def knn(pos_support, pos, k):
    """Dense knn serach
    Arguments:
        pos_support - [B,N,3] support points
        pos - [B,M,3] centre of queries
        k - number of neighboors, needs to be > N

    Returns:
        idx - [B,M,k]
        dist2 - [B,M,k] squared distances
    """
    assert pos_support.dim() == 3 and pos.dim() == 3
    if pos_support.is_cuda:
        raise ValueError("CUDA version not implemented, use pytorch geometric")
    return tpcpu.dense_knn(pos_support, pos, k)
