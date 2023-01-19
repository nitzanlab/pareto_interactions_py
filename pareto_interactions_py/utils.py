from scipy.spatial.distance import squareform, pdist
import numpy as np

def knn_equidistance(X=None, k=5, dist=None, include_self=True):
    """
    Constructs a knn-like adjacency matrix where if distances are equal includes both
    :param X: matrix
    :param k: least number of neighbors
    :return:
    """
    dist = squareform(pdist(X)) if dist is None else dist
    dist = np.around(dist, decimals=4)
    knn = np.zeros_like(dist)
    ar_d = np.arange(dist.shape[0])

    k += 1 if include_self else 0

    for r in ar_d:
        idx_part = np.argsort(dist[r, :]) # np.argpartition(dist[r, :], k)  # smallest k
        idx_same_val = np.where(dist[r, :] == dist[r, idx_part[k]])[0] # this is the k'th
        idx = np.concatenate((idx_part[:k], idx_same_val))
        knn[r, idx] = 1
    # for c in ar_d:
    #     idx = np.argpartition(dist[:, c], k)  # smallest k
    #     knn[idx[:k], c] = 1
    if not include_self:
        np.fill_diagonal(knn, 0)  # include self

    return knn
