import numpy as np


############################################### basic global gradients #################################################

def normx(x):
    """
    :param x: list of points
    :return: normalized gradient
    """
    return (x - x.min()) / (x.max() - x.min())

def no_grad(L):
    """
    :param L: list of points
    :return: gradient"""
    return np.ones_like(L[:, 0])

def linear_grad(L, axis=0, norm=True, reverse=False):
    """
    :param L: list of points
    :param axis: axis to compute gradient
    :param norm: normalize gradient
    :param reverse: reverse gradient
    :return: gradient
    """
    x = L[:, axis]
    if reverse:
        x = x[::-1]
    if norm:
        return normx(x)
    else:
        return x
