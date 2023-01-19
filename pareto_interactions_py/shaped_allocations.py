import numpy as np
import pandas as pd
import random
import itertools
# task space sampling methods


############################################### spatially unaware allocations ##########################################

def get_circum_tasks(n, m, ver, edg, mid, **kwargs):
    """
    Returns task allocation based on ver/edg/mid should be 1/0
    """
    r = 4
    G_emp = np.zeros((0, m))

    # num features
    n_ver = m * ver
    n_edg = m * (m - 1) / 2 * edg
    n_mid = mid

    n_per_feature = int(np.ceil(n / (n_ver + n_edg + n_mid)))

    # vertices
    G_ver = np.tile(np.eye(m), n_per_feature).reshape((m * n_per_feature, m)) if n_ver > 0 else G_emp

    # mid
    G_mid = 1 / m * np.ones((n_per_feature, m)) if n_mid > 0 else G_emp

    # edge
    inc_edge = np.round_(np.arange(n_per_feature) / n_per_feature, r).reshape((-1, 1))
    dec_edge = np.round_(1 - inc_edge, r)
    zer_edge = np.zeros_like(inc_edge)

    w = np.concatenate([inc_edge, dec_edge, zer_edge], 1)

    idx_cols = list(itertools.permutations(np.arange(m)))
    G_edg = np.concatenate([w[:, i] for i in idx_cols]) if n_edg > 0 else G_emp

    # combine
    G = np.concatenate([G_ver, G_edg, G_mid], 0)

    idx = np.arange(G.shape[0])
    random.shuffle(idx)
    G = G[idx, :]
    G = G[:n, :]
    return G

def get_diri_tasks(n, m, alpha, **kwargs):
    G = np.random.dirichlet(np.ones(m) * alpha, size=(n,))
    return G

def get_random_tasks(n, m, sample_desc, params, **kwargs):
    if sample_desc == 'Random':
        G = np.random.random(size=(n, m))
    if sample_desc.startswith('Normal'):
        G = np.random.normal(size=(n, m), **params)
    G = G / G.sum(1).reshape((-1, 1))
    return G

def get_generalists_specialists(n, m, n_generalists, **kwargs):
    n_specialists = n - n_generalists
    generalists = np.full((n_generalists, m), 1 / m)
    specialists = np.tile(np.eye(m), int(np.ceil(n_specialists / m))).reshape(-1, m)[:n_specialists]
    return np.concatenate((generalists, specialists))

################################################# spatially aware allocations ##########################################

# TODO: salt-and-pepper, islets, lines, circumfrence, ...

def get_alternating(n, m, L, spacing, **kwargs):
    # for 1D
    m = int(m)
    spacing = int(spacing)
    n_row = len(np.unique(L[:, 0]))
    n_col = len(np.unique(L[:, 1])) if L.shape[1] == 2 else None
    G = np.tile(np.repeat(np.eye(m).flatten(), spacing).reshape((m, -1)), int(np.ceil(n_row / spacing / m)))[:,
            :n_row].T

    # for 2D
    if n_col:
        G_list = []
        pattern = np.tile(G.flatten(), spacing).reshape((-1, m))

        for i in np.arange(int(np.ceil(n_col / spacing))):
            G_list.append(np.roll(pattern, i, axis=1))

        G = np.concatenate(G_list, 0)[:n, :]
    return G

def get_diagonals(n, m, L, spacing, **kwargs):
    # for 1D
    m = int(m)
    spacing = int(spacing)
    n_row = len(np.unique(L[:, 0]))
    n_col = len(np.unique(L[:, 1])) if L.shape[1] == 2 else None
    G = np.tile(np.repeat(np.eye(m).flatten(), spacing).reshape((m, -1)),
                int(np.ceil(n_row / spacing / m)))[:, :n_row].T

    # for 2D
    if n_col:
        G_list = []
        pattern = G

        for i in np.arange(int(np.ceil(n_col))):
            G_list.append(np.roll(pattern, i, axis=0))

        G = np.concatenate(G_list, 0)#[:n, :]
    return G

# task space
diri_tasks = pd.DataFrame([ {'sample_desc': 'Alpha 0.05', 'alpha': 0.05},
                           {'sample_desc': 'Alpha 0.10', 'alpha': 0.10},
                           {'sample_desc': 'Alpha 0.20', 'alpha': 0.20},
                           {'sample_desc': 'Alpha 0.40', 'alpha': 0.40},
                           {'sample_desc': 'Alpha 0.80', 'alpha': 0.80},
                           {'sample_desc': 'Alpha 1', 'alpha': 1},
                           {'sample_desc': 'Alpha 3', 'alpha': 3}])

diri_tasks['continuity_order'] = np.arange(diri_tasks.shape[0])
diri_tasks['sample_method'] = 'Dirichlet'

# circumfrance task space
circum_tasks = pd.DataFrame([{'sample_desc': 'Vertices', 'ver': 1, 'edg': 0, 'mid': 0},
                             {'sample_desc': 'Vertices-Edges', 'ver': 1, 'edg': 1, 'mid': 0},
                             {'sample_desc': 'Edges', 'ver': 0, 'edg': 1, 'mid': 0}, #?
                             # {'sample_desc': 'Vertices-Center', 'ver': 1, 'edg': 0, 'mid': 1},
                             # {'sample_desc': 'Vertices-Edges-Center', 'ver': 1, 'edg': 1, 'mid': 1},
                             {'sample_desc': 'Edges-Center', 'ver': 0, 'edg': 1, 'mid': 1},
                             {'sample_desc': 'Center', 'ver': 0, 'edg': 0, 'mid': 1}])

circum_tasks['continuity_order'] = np.arange(circum_tasks.shape[0])
circum_tasks['sample_method'] = 'Circum'


# other random task space
random_tasks = pd.DataFrame([{'sample_desc': 'Random', 'params':{}},
                              #{'sample_desc': 'Normal', 'params':{'loc':1/2, 'scale':0.3}},
                              {'sample_desc': 'Normal', 'params':{'loc': 1/2, 'scale': 0.1}},
                              {'sample_desc': 'Normal Conc', 'params':{'loc': 1/2, 'scale': 0.05}}]) # not handling negative samples for now

random_tasks['continuity_order'] = np.arange(random_tasks.shape[0])
random_tasks['sample_method'] = 'Random'

# spacing
alt_tasks = pd.DataFrame([{'sample_desc': 'Spacing 1', 'spacing': 1},
                          {'sample_desc': 'Spacing 2', 'spacing': 2},
                          {'sample_desc': 'Spacing 3', 'spacing': 3}]) # not handling negative samples for now

alt_tasks['sample_method'] = 'Alternate'
alt_tasks['sample_method'] = 'Diagonal'


# entropy-specific
# entropy_tasks
# adding simulations of task spaces with a defined entropy (https://math.stackexchange.com/questions/2248766/randomly-generate-probability-mass-function-with-specific-entropy)
sample_tasks = pd.concat((diri_tasks, circum_tasks, random_tasks, alt_tasks), axis=0, sort=False)

def get_multiple_allocations(n, m, L, tasks_desc_df=sample_tasks):
    task_allocations = {}
    for irow, row in tasks_desc_df.iterrows():
        # sample space
        if row['sample_method'] == 'Circum':
            G = get_circum_tasks(n=n, m=m, **row)
        elif row['sample_method'] == 'Dirichlet':
            G = get_diri_tasks(n=n, m=m, **row)
        elif row['sample_method'] == 'Random':
            G = get_random_tasks(n=n, m=m, **row)
        elif row['sample_method'] == 'Alternate':
            G = get_alternating(n=n, m=m, L=L, **row)
        elif row['sample_method'] == 'Diagonal':
            G = get_diagonals(n=n, m=m, L=L, **row)
        elif row['sample_method'] == 'Generlists-Specialists':
            G = get_generalists_specialists(n=n, m=m, **row) # TODO: feed number of generalists cells
        else:
            print('No function exists for %s' % row['sample_method'])
            continue
        task_allocations[row['sample_method'] + row['sample_desc']] = G
    return task_allocations

if __name__ == '__main__':
    from pareto_opt import get_grid
    import plot_funcs as P
    n = 144
    d = 2
    m = 3
    T = np.eye(m)
    L = get_grid(n, d)
    G = get_diagonals(n, m, L, spacing=2)
    import altair as alt
    alt.renderers.enable('altair_viewer')
    P.plot_physical_space(G, T, L, tit='Physical Space').show()
