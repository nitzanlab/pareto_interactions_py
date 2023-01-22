import numpy as np
import pandas as pd
import altair as alt
from itertools import combinations
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA

pt_size = 100
colors = ['red', 'blue', 'lime', 'yellow', 'purple']

def to_paper(pl, axisFontsize=25, labelFontSize=20, titleFontSize=30, axisTitleFontSize=25):
    """
    Beautify altair chart
    :param pl: altair chart
    """
    pl = pl.configure_view(strokeOpacity=0)
    pl = pl.configure_axis(labelFontSize=axisFontsize, titleFontWeight='normal', titleFontSize=axisTitleFontSize, domainWidth=2, domainColor='black')
    pl = pl.configure_title(fontSize=titleFontSize)
    pl = pl.configure_legend(titleFontSize=labelFontSize, labelFontSize=labelFontSize)
    
    return pl

################################################## Optimization solution ##################################################

def to_2d(T, pca=None, n_components=2):
    """
    Project polytope onto dimension
    :param T: samples x tasks
    :param pca: optional PCA object
    :param n_components: number of components
    :return:
    """
    m = T.shape[1]

    # fit in 2D plot
    if pca is None:
        pca = PCA(n_components=n_components)
        
        if m <= 3:
            pca.fit(T)
        else: # if m > 3, computing a random projection
            tmp = np.random.random((10, m))
            ntmp = (tmp.T / tmp.sum(1)).T
            pca.fit(ntmp)

    coords_cols = ['pc1', 'pc2']
    task_cols = ['task%d' % i for i in np.arange(m)]

    df_coords = pd.DataFrame(pca.transform(T), columns=coords_cols)
    df_color = pd.DataFrame(T, columns=task_cols)
    df = pd.concat((df_coords, df_color), axis=1)
    return df, pca, task_cols



def plot_task_space(G, T, tit='', remove_labels=False):
    """
    Plots cells in task space (uses PCA to show arbitrary task space)
    :param G: cells x tasks
    :param T: tasks x tasks
    :param tit: title
    :param remove_labels: remove axes labels
    :return: plot of cells in task space
    """
    df_T, pca, task_cols = to_2d(T)
    df_G, _, _ = to_2d(G, pca=pca)

    m = G.shape[1]
    if m > len(colors):
        print('Add more colors')

    df_T['color'] = colors[:m]
    df_T['task'] = task_cols

    scale_range = np.max([df_T['pc1'].max() - df_T['pc1'].min(),
                          df_T['pc2'].max() - df_T['pc2'].min()])

    scale_x = alt.Scale(domain=(df_T['pc1'].min(), df_T['pc1'].min() + scale_range))
    scale_y = alt.Scale(domain=(df_T['pc2'].min(), df_T['pc2'].min() + scale_range))
    scale_opacity = alt.Scale(domain=(0, 1), range=(0, 1))

    # task space
    # blend: https://developer.mozilla.org/en-US/docs/Web/CSS/mix-blend-mode
    axis_kwargs = {'title': None} if remove_labels else {}
    axis = alt.Axis(grid=False, labels=False, ticks=False, domain=False, **axis_kwargs)

    fold = 10
    d_x = 15 * fold
    d_y = 1 if m <= 2 else 15
    d_y = fold * (d_y)

    kwargs_y = {'y': alt.Y('pc2', scale=scale_y, axis=axis)} if m >= 3 else {}
    pts = alt.Chart(df_G, width=d_x, height=d_y, title=tit).mark_circle(size=pt_size, blend=alt.Blend('multiply'),
                                                 shape='circle').encode(x=alt.X('pc1', scale=scale_x, axis=axis), **kwargs_y)
    
    # points
    charts = []
    for i in np.arange(m):
        charts.append(pts.encode(color=alt.value(colors[i]),
                                 opacity=alt.Opacity(task_cols[i], scale=scale_opacity, legend=None)))
    p_G = alt.layer(*charts)
    base_ln = alt.Chart().mark_line(strokeWidth=3).encode(color=alt.value('black'),
                                                          opacity=alt.value(1),
                                                          x=alt.X('pc1', scale=scale_x, axis=axis),
                                                          **kwargs_y)
    p_T_list = []
    for i in combinations(np.arange(m), 2):
        p_T_list.append(base_ln.properties(data=df_T.iloc[list(i)]))
    p_T = alt.layer(*p_T_list)

    p_task = alt.layer(p_T, p_G).resolve_scale(x='shared', y='shared')
    return p_task



def plot_physical_space(G, T, L, tit='Physical Space', remove_labels=False):
    """
    Plot cells in their physical location and color by their task allocation
    :param G: cells x tasks
    :param T: tasks x tasks
    :param L: cells x dims
    :param tit: title
    :param remove_labels: remove axes labels
    :return: plot of cells in task space
    """
    _, pca, task_cols = to_2d(T)
    df_G, _, _ = to_2d(G, pca=pca)

    m = G.shape[1]
    d = L.shape[1]

    # plot locations
    xyz = ['x', 'y', 'z']
    df_L = pd.DataFrame(L).rename(columns={i: xyz[i] for i in np.arange(d)})
    df = pd.concat((df_G, df_L), axis=1)

    fold = 10
    d_x = (df_L['x'].max() - df_L['x'].min() + 1) * fold
    d_y = 1 if df_L.shape[1] < 2 else (df_L['y'].max() - df_L['y'].min() + 1)
    d_y = fold*(d_y)
    axis_kwargs = {'title': None} if remove_labels else {}
    axis = alt.Axis(grid=False, labels=False, ticks=False, domain=False, **axis_kwargs)

    pts_L = alt.Chart(df, width=d_x, height=d_y).mark_circle(size=pt_size)
    if d == 1:
        pts_L = pts_L.encode(x=alt.X('x', axis=axis))
    elif d == 2:
        pts_L = pts_L.encode(x=alt.X('x', axis=axis),
                             y=alt.Y('y', axis=axis))
    else:
        print('Does not handle dim > 2')

    scale_opacity = alt.Scale(domain=(0, 1), range=(0, 1))

    charts_L = []
    for i in np.arange(m):
        charts_L.append(pts_L.encode(color=alt.value(colors[i]),
                                     opacity=alt.Opacity(task_cols[i], scale=scale_opacity, legend=None)))
    p_L = alt.layer(*charts_L)
    return p_L



def plot_sol(G, T, L, n_neigh=None, obj=None, tit='', vconcat=True, remove_labels=False):
    """
    Plot both task and physical organization of cells
    :param G: cells x tasks
    :param T: tasks x tasks
    :param L: cells x dims
    :param n_neigh: num of neighbors (used only for title)
    :param obj: objective function (used only for title)
    :param tit: title
    :param vconcat: whether to concat plot vertically or horizontally
    :param remove_labels: remove axes labels
    :return: plot of cells in task space
    """
    p_task = plot_task_space(G, T, remove_labels=remove_labels)
    p_L = plot_physical_space(G, T, L, remove_labels=remove_labels)
    n, m = G.shape
    d = L.shape[1]
    if tit is None:
        tit = ''
    else:
        if tit == '':
            tit += 'num cells: %d, num tasks: %d, num dims: %d' % (n, m, d)
            tit += ', num neigh: %d' % n_neigh if n_neigh else ''
        tit += ', score: %.02f' % -obj(G.flatten()) if obj else ''

    if vconcat:
        p = alt.vconcat(p_task, p_L, title=tit, center=True).resolve_scale(size='independent')
    else:
        p = alt.hconcat(p_task, p_L, title=tit, center=True).resolve_scale(size='independent')

    return to_paper(p)
    

################################################## Correlation analysis ##################################################



def comp_task_phys_bins(df, nbin=10, xbins=None, ybins=None, r=2, xcol='phys dist', ycol='task dist'):
    """
    Partitions values of two columns into bins
    :param df: dataframe containing: `phys dist` and `task dist`
    :param nbin: number of bins to split to
    :param xbin: optional pre-determined bins for xcol
    :param ybin: optional pre-determined bins for ycol
    :param xcol: name of xcol
    :param ycol: name of ycol
    :param r: round values to r order
    """
    def to_bins(col, bins, quantile=0.99):
        """
        Divide values in column in dataframe to bins
        """
        if col not in df.columns:
            print(f'Dataframe is missing column {col}')
            return 
        col_s = col + 'standardized'
        df[col_s] = df[col]
        
        # remove large distance outliers
        max_lim = df[col_s].quantile(quantile)
        df.loc[df[col_s] > max_lim, col_s] = max_lim
        df[col_s] = df[col_s] / max_lim
        
        # round bin threshold
        bins = np.linspace(0, 1, nbin) if bins is None else bins
        bins = np.round(bins, r) if r else bins

        # cut and represent by mid value
        df[col + ' bined int'] = pd.cut(df[col_s], bins=bins, right=False)
        to_mid = {k: np.round(k.mid, r) for k in df[col + ' bined int'].cat.categories}
        df[col + ' bined'] = df[col + ' bined int'].apply(lambda x: to_mid[x])
        return bins

    xbins = to_bins(xcol, xbins)
    ybins = to_bins('task dist', ybins)

    cols_b = [f'{xcol} bined', f'{ycol} bined']
    df_stats = df.groupby(cols_b).size().reset_index()
    df_stats.rename(columns={0: 'num pairs'}, inplace=True)

    df_stats['pair fraction'] = df_stats['num pairs'] / df.shape[0]

    return df_stats, xbins, ybins



def plot_task_phys_bins(df_stats, tit='', add_legend=True, legendtickCount=5, tickCount=1):
    """
    Plot bins of task and physical distances
    :param df_stats: dataframe containing: `phys dist bined` and `task dist bined`
    :param tit: title
    :param add_legend: add legend
    :param legendtickCount: number of bin sizes described in legend
    :param tickCount: number of ticks in x,y axes
    """
    axis = alt.Axis(grid=False, tickCount=tickCount)
    legend = alt.Legend(tickCount=legendtickCount) if add_legend else None
    p = alt.Chart(df_stats, title=tit).mark_circle(aspect=True).encode(x=alt.X('phys dist bined:Q', title='Physical distance', axis=axis),
                                                            y=alt.Y('task dist bined:Q', title='Task distance', axis=axis),
                                                            color=alt.value('black'),
                                                            size=alt.Size('pair fraction:Q',
                                                                          scale=alt.Scale(range=[0, 1500], domain=[0, 0.1]),
                                                                          legend=legend),
                                                            opacity=alt.value(1))
    
    return to_paper(p)


def plot_task_phys(X_task, X_spatial, comp_null=False, nbin=10, r=2, n_shuff=1000, verbose=True, vconcat=True, tit=None, add_legend=True, **kwargs):
    """
    Plot task and physical distances
    :param X_task: task allocation per cell (cell x tasks)
    :param X_spatial: spatial location (cell x dim)
    :param comp_null: whether to compute null distribution
    """

    # compute dists
    task_dist = pdist(X_task)
    phys_dist = pdist(X_spatial)
    df = pd.DataFrame({'task dist': task_dist, 'phys dist': phys_dist})
    n_spatial = X_spatial.shape[0]

    # bin and plot data
    df_stats, xbins, ybins = comp_task_phys_bins(df, nbin, r=r)

    corr = np.corrcoef(df['task dist'], df['phys dist'])[0,1]

    if verbose:
        print('corr: %.02f' % corr)

    corr_null_list = []
    if comp_null:
        # generate null
        for _ in np.arange(n_shuff):
            idx = np.random.permutation(n_spatial) # TODO: substituted
            phys_dist_perm = pdist(X_spatial[idx, :])
            df_null = pd.DataFrame({'task dist': task_dist, 'phys dist': phys_dist_perm})
            corr_null_list.append(np.corrcoef(df_null['task dist'], df_null['phys dist'])[0,1])

        pval = (corr <= corr_null_list).mean()
        
        print('pval: %.05f' % pval)

    # plot dist correlation plot
    pval_text = '<%.2E' % (1 / n_shuff) if pval == 0 else '%.2E' % pval
    alt_tit = '%.02f, %s' % (corr, pval_text) if comp_null else '%.02f' % corr
    tit = tit if tit is not None else alt_tit
    
    p = plot_task_phys_bins(df_stats, tit=tit, **kwargs)

    return p
    