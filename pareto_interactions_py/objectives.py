import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist, pdist, squareform

epsilon = 1e-8
D_METRIC = 2 # was running with 2
D_NORM = 2
D_POWER = 2

def plot_stats(G, D, P, H=None):
    fig, ax = plt.subplots(1, 2)
    ax[0].scatter(D.flatten(), P.flatten());
    ax[0].set_xlabel('Distance D'); ax[0].set_ylabel('performance P');

    if H is not None:
        ax[1].scatter(P.flatten(), H.flatten());
        ax[1].set_xlabel('performance P'); ax[1].set_ylabel('neighbors weighing H');

    plt.show()

def obj_type(g, n, m, T, a0, grads=None, plot=False, D_metric=D_METRIC, D_power=D_POWER):
    """
    Objective just for tasks
    :param g: flattened G
    :param n: num cells
    :param m: num tasks
    :param T: tasks x tasks - optimal points of tasks
    :param a0: max performance on tasks
    :param grads: cells x tasks weight for a task for each cell's location
    :return: negative fitness objective
    """
    G = g.reshape((n, m))
    D = cdist(G, T, metric='minkowski', p=D_metric)
    P = (a0 - D ** D_power) / a0  # dist cell to task, dij, cell x task

    grads = np.ones((n, m)) if grads is None else grads
    gradP = np.multiply(grads, P)

    if plot:
        plot_stats(G, D, P)
    return -np.prod(np.sum(gradP, 0))


def obj_neigh_avg_performance(g, n, m, T, a0, N, grads=None, plot=False, D_metric=D_METRIC):
    """
    Objective considering neighbors contribution

        F = \prod_taskj \sum_celli Hji Pji
        Hji = ((a0 - (N @ P)) / a0)

    :param g: flattened G
    :param n: num cells
    :param m: num tasks
    :param T: tasks x tasks - optimal points of tasks
    :param a0: max performance on tasks
    :param N: normalized neighbors matrix
    :return: negative fitness objective
    """
    G = g.reshape((n, m))
    D = cdist(G, T, metric='minkowski', p=D_metric)
    P = (a0 - D ** 2)  # dist cell to task, dij, cell x task
    grads = np.ones((n, m)) if grads is None else grads
    gradP = np.multiply(grads, P) #?
    H = ((a0 - (N @ gradP)) / a0)  # cell x task

    if plot:
        plot_stats(G, D, P, H)

    return -np.prod(np.sum(H * gradP, 0))
    # return -np.prod(np.sum(H * P, 0))

def obj_neigh_avg_arch(g, n, m, T, a0, N, grads=None, plot=False, D_metric=D_METRIC):
    """
    Objective considering neighbors contribution

        F = \prod_taskj \sum_celli Hji Pji
        Hji = ((a0 - P(N @ G)) / a0)

    :param g: flattened G
    :param n: num cells
    :param m: num tasks
    :param T: tasks x tasks - optimal points of tasks
    :param a0: max performance on tasks
    :param N: normalized neighbors matrix
    :return: negative fitness objective
    """
    G = g.reshape((n, m))
    WG = N @ G
    D = cdist(WG, T, metric='minkowski', p=D_metric)
    P = a0 - D ** 2  # dist cell to task, dij, cell x task
    grads = np.ones(n, m) if grads is None else grads
    gradP = np.multiply(grads, P)  # ?
    H = ((a0 - (gradP)) / a0)  # cell x task

    if plot:
        plot_stats(G, D, P, H)

    return -np.prod(np.sum(H * gradP, 0))
    # return -np.prod(np.sum(H * P, 0))


def obj_neigh_avg_performance(g, n, m, T, a0, N, grads=None, plot=False, D_metric=D_METRIC):
    """
    Objective considering neighbors contribution

        F = \prod_taskj \sum_celli Hji Pji
        Hji = ((a0 - (N @ P)) / a0)

    :param g: flattened G
    :param n: num cells
    :param m: num tasks
    :param T: tasks x tasks - optimal points of tasks
    :param a0: max performance on tasks
    :param N: normalized neighbors matrix
    :return: negative fitness objective
    """
    G = g.reshape((n, m))
    D = cdist(G, T, metric='minkowski', p=D_metric)
    P = (a0 - D ** 2)  # dist cell to task, dij, cell x task
    grads = np.ones((n, m)) if grads is None else grads
    gradP = np.multiply(grads, P) #?
    H = ((a0 - (N @ gradP)) / a0)  # cell x task

    if plot:
        plot_stats(G, D, P, H)

    return -np.prod(np.sum(H * gradP, 0))
    # return -np.prod(np.sum(H * P, 0))


def obj_neigh_avg_performance01(g, n, m, T, a0, N, beta, plot=False, D_metric=D_METRIC, **kwargs):
    """
    Objective considering neighbors contribution
    :param g: flattened G
    :param n: num cells
    :param m: num tasks
    :param T: tasks x tasks - optimal points of tasks
    :param a0: max performance on tasks
    :param N: normalized neighbors matrix
    :return: negative fitness objective
    """
    G = g.reshape((n, m))
    D = cdist(G, T, metric='minkowski', p=D_metric)
    P = (a0 - D ** 2) / a0  # dist cell to task, dij, cell x task
    H = (1 - (N @ P))  # cell x task

    if plot:
        plot_stats(G, D, P, H)

    return -np.prod(np.sum(H * P, 0))

def plot_h(h):
    N = np.ones((1, 1))
    P = np.linspace(0, 1, 50).reshape((1,-1))
    plt.scatter(P.flatten(), h(N, P).flatten());
    plt.xlabel('avg performance P');
    plt.ylabel('neighbors weighing H');
    plt.show()

def obj_global_local(g, n, m, T, a0, N, beta, grads=None, plot=False, D_norm=D_NORM, D_power=D_POWER,
                     h=None, norm_performance=True):
    """
    Objective considering neighbors contribution
    :param g: flattened G
    :param n: num cells
    :param m: num tasks
    :param T: tasks x tasks - optimal points of tasks
    :param a0: max performance on tasks
    :param N: normalized neighbors matrix
    :return: negative fitness objective
    """

    G = g.reshape((n, m))
    D = cdist(G, T, metric='minkowski', p=D_norm)

    if norm_performance:
        P = (a0 - D ** D_power) / a0  # dist cell to task, dij, cell x task
    else:
        P = (a0 - D ** D_power)  # dist cell to task, dij, cell x task

    h = (lambda N,P: (1 - (N @ P))) if h is None else h
    H = h(N, P)

    # h = (lambda N,P: (1 - (N @ P))) if h is None else h
    # H = h(N, P)


    if plot:
        plot_h(h)
        plot_stats(G, D, P, H)

    grads = np.ones((n, m)) if grads is None else grads

    return -np.prod(np.sum(np.multiply(((1-beta) * grads + beta * H), P), 0))

def obj_miris(g, n, m, T, a0, N, beta, grads=None, plot=False, D_norm=D_NORM, D_power=D_POWER,
                     h=None, norm_performance=True):
    """
    Objective considering neighbors contribution
    :param g: flattened G
    :param n: num cells
    :param m: num tasks
    :param T: tasks x tasks - optimal points of tasks
    :param a0: max performance on tasks
    :param N: normalized neighbors matrix
    :return: negative fitness objective
    """

    G = g.reshape((n, m))
    D = cdist(G, T, metric='minkowski', p=D_norm)

    P = (a0 - D ** D_power) / a0

    h = (lambda N,P: (1 - (N @ P))) if h is None else h
    H = h(N, P)

    if plot:
        plot_h(h)
        plot_stats(G, D, P, H)

    grads = np.ones((n, m)) if grads is None else grads

    return -np.prod(np.sum(np.multiply(((1-beta) * grads + beta * H), P), 0))







obj_neigh_default = obj_global_local

#obj_neigh_avg_performance01
# obj_neigh_default = obj_neigh_avg_performance01 # obj_global_local
