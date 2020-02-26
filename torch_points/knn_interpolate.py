import torch
from .knn import knn

def knn_interpolate(x, pos, pos_support, k):
    """ KNN interpolation for dense data

    Parameters
    ----------
    x : (B, C, n) tensor of known features
    pos : (B, n, 3) tensor of positions of known features
    pos_support : (B, m, 3) tensor of position of unknown  features (generally m > n)

    Returns
    -------
    (B, C, m) interpolated features
    """
    
    knn_idx, knn_dist = knn(pos_support, pos, k)


