import numpy as np
import itertools
import pandas as pd
from scipy.spatial.distance import cdist

############################################### basic global gradients #################################################

def normx(x):
    return (x - x.min()) / (x.max() - x.min())

def no_grad(L):
    return np.ones_like(L[:, 0])

def linear_grad(L, axis=0, norm=True, reverse=False):
    x = L[:, axis]
    if reverse:
        x = x[::-1]
    if norm:
        return normx(x)
    else:
        return x

def radial_grad(L, center=[0,0], metric='Euclidean', norm=True):
    center = np.array(center).reshape((1,-1))
    x = cdist(L, center).flatten()
    if norm:
        return normx(x)
    else:
        return x


############################################### uncover global gradients ###############################################

def polyfit2d(x, y, z, order=3):
    """
    Fit a value (task levels) to 2d space
    :param x:
    :param y:
    :param z:
    :param order:
    :return:
    """
    ncols = (order + 1) ** 2
    G = np.zeros((x.size, ncols))
    ij = itertools.product(range(order + 1), range(order + 1))

    coeff_list = []
    for k, (i, j) in enumerate(ij):
        G[:, k] = x ** i * y ** j
        coeff_list.append({'x deg': i, 'y deg': j})

    coeff_df = pd.DataFrame(coeff_list)
    m, r, rank, s = np.linalg.lstsq(G, z)
    coeff_df['coeff'] = m
    return m, coeff_df


def polyval2d(x, y, m):
    order = int(np.sqrt(len(m))) - 1
    ij = itertools.product(range(order + 1), range(order + 1))
    z = np.zeros_like(x)
    for a, (i, j) in zip(m, ij):
        z += a * x ** i * y ** j
    return z
