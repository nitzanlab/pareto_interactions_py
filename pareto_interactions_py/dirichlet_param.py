import numpy as np
import pandas as pd
from .pareto_opt import get_neigh_mat, get_grid, obj_neigh


def obj_neigh_by_dirichlet(alpha, n, m, T, a0, N, **kwargs):
    """
    Compute the fitness given the concentration parameter describing how specialists-generalists are the cells.
    :param alpha: dirichlet concentration parameter, alpha > 0
    :return:
    """
    # sample with dirichlet cells task allocation
    G = np.random.dirichlet(np.ones(m) * alpha, shape=(n,))
    # G = random.dirichlet(rng, np.ones((1, m)) * alpha, shape=(n,))

    # function to optimize
    mF = obj_neigh(G, n, m, T, a0, N) # minus F
    return mF


if __name__ == '__main__':


    setups = {'1d_2neighs_2tasks': {'d': 1, 'm': 2, 'n_neigh': 2},
              '2d_4neighs_3tasks': {'d': 2, 'm': 3, 'n_neigh': 4},
              '2d_16neighs_3tasks': {'d': 2, 'm': 3, 'n_neigh': 16}}
    n = 100

    for _, v in setups.items():
        v['n'] = n
        v['a0'] = v['m']

        # ini
        v['L'] = get_grid(v['n'], v['d'])  # np.arange(n).reshape((-1,1))
        v['N'] = get_neigh_mat(v['L'], v['n_neigh'], is_cyclic=True)
        v['T'] = np.eye(v['m'])

    # Testing
    # 2 neighbors, 1 d, expect
    # small alpha = specialists, big alpha = generalists
    alphas = np.arange(0.1, 1, 0.1)

    res = []
    for k, v in setups.items():
        for alpha in alphas:
            res.append({'setup': k,
                        'alpha': alpha,
                        'obj_val': obj_neigh_by_dirichlet(alpha, **v)})
    print(pd.DataFrame(res))
