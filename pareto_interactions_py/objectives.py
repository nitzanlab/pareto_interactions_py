import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

epsilon = 1e-8
D_METRIC = 2
D_NORM = 2
D_POWER = 2

def plot_stats(G, D, P, H=None):
    """
    Plot the stats of the objective
    :param G: cells' task allocation (cells x tasks)
    :param D: cells' distance from archetype expression (cells x tasks)
    :param P: (self) performance of a cell in a task (cells x tasks)
    :param H: interaction term, neighbors' affect on a cell in task (cells x tasks)
    """
    ncols = 2 if H is not None else 1
    _, ax = plt.subplots(1, ncols, figsize=(5*ncols, 5))

    ax[0].scatter(D.flatten(), P.flatten())
    ax[0].set_xlabel('distance D') 
    ax[0].set_ylabel('performance P')
    ax[0].set_title('Performance in task verses distance from its archetype')
    
    if H is not None:
        ax[1].scatter(P.flatten(), H.flatten())
        ax[1].set_xlabel('performance P') 
        ax[1].set_ylabel('neighbors weighing H')
        ax[1].set_title('Neighbor weighing of task verses the cell\'s performance in it')

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

    
    if plot:
        plot_stats(G, D, P, H)

    grads = np.ones((n, m)) if grads is None else grads

    return -np.prod(np.sum(np.multiply(((1-beta) * grads + beta * H), P), 0))



obj_neigh_default = obj_global_local
