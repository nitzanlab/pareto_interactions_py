import numpy as np
import pandas as pd
from .objectives import obj_type, obj_neigh_default
from scipy.optimize import minimize
from scipy.optimize.optimize import approx_fprime
# from grispy import GriSPy
from sklearn.neighbors import kneighbors_graph
from .shaped_allocations import get_multiple_allocations
import igraph as ig

# n - num cells
# m - num tasks
# d - num dims

MAXITER = 1000
epsilon = 1e-16

############################################### allocation initialization ##############################################

def get_ini_rnd(n, m):
    """
    Random initialization of archetypes for n cell, m task components
    """
    g0 = np.random.rand(n, m)  # cell x task
    g0 = (g0.T / np.sum(g0, 1).T).T
    return g0.flatten()

def get_ini_unif(n, m, eps=0):
    """
    Uniform initialization of archetypes for n cell, m task with eps noise
    """
    g0 = np.ones((n, m))  # cell x task
    g0 = (g0.T / np.sum(g0, 1)).T
    g0 += np.random.normal(0, eps, (n, m))
    if np.any(g0 < 0):
        print('Negative values in ini. Taking abs value')
        g0 = np.abs(g0)
    g0 = (g0.T / np.sum(g0, 1)).T
    return g0.flatten()



def cons_sum_to_one(g, n, m):
    """
    Check constraint: returns 0 for cells whose allocation sums to 1
    :param g: flattened G (nxm) archetype matrix
    :param n: num cells
    :param m: num tasks
    :return: cell x 1 for each cell 0 if tasks sum to 1
    """
    G = g.reshape((n, m))
    sG = np.sum(G, 1) # np.linalg.norm(G, axis=1)
    return -sG + np.ones_like(sG)


def get_grid(n, d):
    """
    Constructs a squarish grid
    :param n: num of cells
    :param d: num of dimensions
    :return: n x d specification of cell locations
    """
    #     t = np.arange(0, 1., 1 / np.power(n, 1 / d))
    t = np.arange(0, int(np.ceil(np.power(n, 1 / d))))
    w = [t, ] * d
    # generate d coordinates
    M = np.meshgrid(*w)
    fM = tuple(M[i].flatten() for i in np.arange(len(M)))
    L = np.stack(fM)[:, 0:n].T
    return L


def get_inrange_mat(L, n_neigh, is_cyclic=False): # TODO: relevant only for grid
    """
    Currently, only for full grid, sets as neighbor cells within order range from cell
    :param L: location of each cell (n x d)
    :param range: spatial range of neighbors
    :param is_cyclic: is the grid cyclic
    :return: 
    """
    nrow = len(np.unique(L[:, 0]))
    n = nrow
    dim = [nrow]
    if L.shape[1] == 2:
        ncol = len(np.unique(L[:, 1]))
        n = nrow * ncol
        dim = [nrow, ncol]
    g = ig.Graph.Lattice(dim=dim, circular=is_cyclic)
    neighss = g.neighborhood(np.arange(n), order=n_neigh)
    N = np.zeros((n, n))
    for i, neighs in enumerate(neighss):
        N[i, neighs] = 1
    N = N - np.eye(n)
    sN = (N.T / np.sum(N, 1)).T  # cell x neighbors
    return sN


def get_neigh_mat(L, n_neigh, is_cyclic=False):
    """
    Computes neighbor matrix
    :param L: location of each cell (n x d)
    :param n_neigh: number of neighbors
    :param is_cyclic: is the grid cyclic
    :return: cell x cell normalized weight matrix
    """
    n, d = L.shape
    n_neigh = int(n_neigh)

    if n_neigh >= n:
        print('Can have %d (n-1) neighbors at most' % (n-1))
        return
    if n_neigh == (n - 1):
        N = np.ones((n,n)) - np.eye(n)
    # elif is_cyclic:
    #     N = get_neigh_mat_cyclic(L, n_neigh)
    else:
        N = get_neigh_mat_noncyclic(L, n_neigh)

    sN = (N.T / np.sum(N, 1)).T  # cell x neighbors
    return sN


def get_neigh_mat_noncyclic(L, n_neigh):
    """
    Computes neighbor matrix
    :param L: location of each cell (n x d)
    :param n_neigh: number of neighbors
    :return: cell x cell neighbor matrix
    """
    N = kneighbors_graph(L, n_neigh, mode='connectivity', include_self=False).todense()
    return N


# def get_neigh_mat_cyclic(L, n_neigh):
#     """
#     Computes neighbor matrix
#     :param L: location of each cell (n x d)
#     :param n_neigh: number of neighbors
#     :return: cell x cell normalized weight matrix
#     """
#     n, d = L.shape
#     N = np.zeros((n, n))
#
#     val = (0, 360)
#     periodic = {int(i): val for i in np.arange(d)}
#     gsp = GriSPy(L, periodic=periodic)
#     near_dist, near_ind = gsp.nearest_neighbors(L, n=n_neigh)
#     for i, idx in enumerate(near_ind):
#         N[i, idx] = 1
#     return N


def callback(x, obj, res):
    """
    Retrieve path and gradient from optimization
    :param x: current state
    :param obj: objective
    :param res: list of gradients with respect to objective
    """
    res.append({'g': x, 'grad_approx': approx_fprime(x, obj, 1E-10)})


def solve_opt(n, m, d, n_neigh, asrange=True, beta=1, is_cyclic=False, get_path=False, g0=None, L=None, N=None, a0=None,
              fgrads=None,
              obj_neigh=None,
              test=True, maxiter=MAXITER, plot=False, **kwargs):
    """
    Solves optimization
    :param n: number of cells
    :param m: number of tasks
    :param d: number of dimensions
    :param n_neigh: number of neighbors
    :param asrange: use number of neighbors as range, order of neighbors to include in neighborhood
    :param beta: weight of interactions; beta * interaction_weight + (1-beta) * external_weight
    :param is_cyclic: is the grid cyclic
    :param get_path:
    :param g0: initial task allocation
    :param L: inserted spatial space
    :param a0: value of each task
    :param fgrads: array of functions for each task
    :param test: test against alternative task allocations


    :return:
    G - cells x tasks - for each cell, the component for each task
    T - tasks x tasks - optimal phenotypes
    L - location of each cell (n x d)
    F - value of objective
    """
    T = np.eye(m)  # G_i*

    g0 = get_ini_rnd(n, m) if g0 is None else g0.flatten()

    L = get_grid(n, d) if L is None else L
    n, d = L.shape

    # neighbor matrix
    no_neigh = n_neigh < 1
    if N is None:
        if no_neigh:
            N = None
        elif asrange:
            N = get_inrange_mat(L, n_neigh=n_neigh, is_cyclic=is_cyclic)
        else:
            N = get_neigh_mat(L, n_neigh=n_neigh, is_cyclic=is_cyclic)


    a0_val = 2 # diameter of 2-simplex is 1, and is 2 in 3+-simplex
    a0 = a0_val if a0 is None else a0
    # a0 = np.full((n, m), a0_val) if a0 is None else np.full((n, m), a0)

    # constraints
    b = (0.0, 1.0)
    bnds = [(b)] * n * m
    cons = ([
        {'type': 'eq', 'fun': lambda g: cons_sum_to_one(g, n, m)},
    ])

    # compute gradients per location
    grads = np.ones((n, m))
    if fgrads is not None:
        for igrad, fgrad in enumerate(fgrads):
            grads[:, igrad] = fgrad(L)

    # optimization
    obj_neigh = obj_neigh_default if obj_neigh is None else obj_neigh
    obj = (lambda g: obj_type(g=g, n=n, m=m, T=T, a0=a0, grads=grads, **kwargs)) if no_neigh \
        else (lambda g: obj_neigh(g=g, n=n, m=m, T=T, a0=a0, N=N, beta=beta, grads=grads, **kwargs))
    res = []
    callback(g0, obj, res)
    sol = minimize(obj,
                   g0,
                   # method='nelder-mead', # slow, doesn't optimize well
                   method='SLSQP', # fast, returns cells out of bounds, default when unspecified
                   # method='trust-constr', # slow
                   bounds=bnds,
                   constraints=cons,
                   options={'maxiter': maxiter},
                   # callback=lambda x: callback(x, obj, res)
                   )
    g = sol.x
    G = g.reshape((n, m))
    F = obj(sol.x)

    print(sol.message)

    if plot:
        if no_neigh:
            obj_type(g=g, n=n, m=m, T=T, a0=a0, grads=grads, plot=plot)
        else:
            obj_neigh(g=g, n=n, m=m, T=T, a0=a0,  N=N, beta=beta, grads=grads, plot=plot, **kwargs)

    if test:
        task_allocations = get_multiple_allocations(n, m, L)

        # G_desc = 'CircumCenter'
        # G_alt = task_allocations[G_desc]
        # obj(G_alt.flatten())
        # obj(G.flatten())

        for G_desc, G_alt in task_allocations.items():
            F_alt = obj(G_alt.flatten())
            if F_alt < F:
                print('Not optimized - %s task allocation bits optimized allocation' % G_desc)

    # return path
    if get_path:
        res = pd.DataFrame(res)
        # res['func val'] = res['g'].apply(obj)

        # into df
        df_list = []
        for irow, row in res.iterrows():
            df_g = pd.DataFrame(row['g'].reshape((n, m))).rename(columns={i: 'G_%d' % i for i in np.arange(m)})
            df_grad = pd.DataFrame(row['grad_approx'].reshape((n, m))).rename(
                columns={i: 'grad_approx_%d' % i for i in np.arange(m)})
            df = pd.concat((df_g, df_grad), 1)
            df['iter'] = irow
            df_list.append(df)

        df = pd.concat(df_list)
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'cell_id'}, inplace=True)

        return G, T, L, obj, df

    # G = (G.T / G.sum(1)).T
    return G, T, L, obj



def test_solve_opt():
    # import scanpy as sc
    import altair as alt
    from plot_funcs import plot_sol
    from global_grads import no_grad, linear_grad
    n = 100  # num of cells
    m = 3 # num of tasks
    d = 2  # num of dims
    n_neigh = 8  # num neigh

    def h_exp_inverse(N, P, rate=1.5):
        return np.exp(- rate * (N @ P))

    G, T, L, obj = solve_opt(n, m, d, n_neigh, beta=1, test=False, is_cyclic=False, a0=2, h=h_exp_inverse, asrange=True)
    # print(G.sum(1))
    # n_neigh = 2
    # n_2d = 100

    # from sklearn.neighbors import NearestNeighbors
    #
    # def get_neigh_mat(L, n_neigh):
    #     nbrs = NearestNeighbors(n_neighbors=n_neigh + 1, algorithm='ball_tree').fit(L)
    #     N = nbrs.kneighbors_graph(L).toarray()
    #     N[np.diag_indices_from(N)] = 0
    #     sN = (N.T / np.sum(N, 1)).T
    #     return sN
    #
    # # random scatter
    # L_rand = np.random.random((n, 2)) * 15
    # N_rand = get_neigh_mat(L_rand, 4 * n_neigh)
    #
    # import matplotlib.pyplot as plt
    # plt.imshow(N_rand)
    # plt.show()
    #
    # G, T, L, obj = solve_opt(n, m, d, n_neigh, L=L_rand, N=N_rand, beta=1, test=False, is_cyclic=False, a0=2)

    alt.renderers.enable('altair_viewer')
    plot_sol(G, T, L, n_neigh, obj).show()

    pass

if __name__ == '__main__':
    test_solve_opt()

