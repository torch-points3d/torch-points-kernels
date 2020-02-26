import torch_points.points_cpu as tpcpu

def knn(pos_support, pos, k):
    """ Dense knn serach
    Arguments:
        pos_support - [B,N,3] support points
        pos - [B,M,3] centre of queries
        k - number of neighboors, needs to be > N

    Returns:
        idx - [B,M,k]
        dist2 - [B,M,k] squared distances
    """
    return tpcpu.dense_knn(pos_support, pos, k)