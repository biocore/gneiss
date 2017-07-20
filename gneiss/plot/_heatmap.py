# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from gneiss.plot._dendrogram import SquareDendrogram
from gneiss.util import match_tips, NUMERATOR, DENOMINATOR


def heatmap(table, tree, mdvar, highlights=None, cmap='viridis',
            linewidth=0.5, grid_col='w', grid_width=2, highlight_width=0.02,
            figsize=(5, 5), **kwargs):
    """ Creates heatmap plotting object

    Parameters
    ----------
    table : pd.DataFrame
        Contain sample/feature labels along with table of values.
        Rows correspond to samples, and columns correspond to features.
    tree: skbio.TreeNode
        Tree representing the feature hierarchy.
    highlights: pd.DataFrame or dict of tuple of str
        List of internal nodes in the tree to highlight.
        Each internal node must contain two colors, one for the left
        subtree and the other for the right subtree highlight.
        The first color will always correspond to the left subtree,
        and the second color will always correspond to the right subtree.
    cmap : str
        Specifies the matplotlib colormap for the heatmap (default='viridis')
    linewidth : int
        Width of dendrogram lines.
    mdvar: pd.Series
        Metadata values for samples.  The index must correspond to the
        index of `table`.
    highlight_width : int
        Width of highlights. (default=0.02)
    grid_col: str
        Color of vertical lines for highlighting sample metadata.
        (default='w')
    grid_width: int
        Width of vertical lines for highlighting sample metadata.
        (default=2)
    figsize: tuple of int
        Species (width, height) for figure. (default=(5, 5))
    **kwargs : dict
        Arguments to be passed into the heatmap plotting.

    Returns
    -------
    matplotlib.pyplot.figure
        Matplotlib figure object

    Note
    ----
    The highlights parameter assumes that the tree is bifurcating.
    """
    # match the tips
    dendrogram_width = 20

    table, tree = match_tips(table, tree)
    table = table.T
    # get edges from tree
    t = SquareDendrogram.from_tree(tree)
    t = _tree_coordinates(t)
    pts = t.coords(width=dendrogram_width, height=table.shape[0])
    edges = pts[['child0', 'child1']]
    edges = edges.dropna(subset=['child0', 'child1'])
    edges = edges.unstack()
    edges = pd.DataFrame({'src_node': edges.index.get_level_values(1),
                          'dest_node': edges.values})

    edge_list = []
    for i in edges.index:
        src = edges.loc[i, 'src_node']
        dest = edges.loc[i, 'dest_node']
        sx, sy = pts.loc[src].x, pts.loc[src].y
        dx, dy = pts.loc[dest].x, pts.loc[dest].y

        edge_list.append(
            {'x0': sx, 'y0': sy, 'x1': sx, 'y1': dy}
        )
        edge_list.append(
            {'x0': sx, 'y0': dy, 'x1': dx, 'y1': dy}
        )
    edge_list = pd.DataFrame(edge_list)

    # now plot the stuff
    fig = plt.figure(figsize=figsize, facecolor='white')
    plt.rcParams['axes.facecolor'] = 'white'
    xwidth = 0.2
    top_buffer = 0.1
    height = 0.8
    cbar_pad = 0.95

    # dendrogram axes on the left side
    [axm_x, axm_y, axm_w, axm_h] = [0, top_buffer, xwidth, height]

    # create a split for the highlights
    if highlights is not None:
        h = len(highlights)
    else:
        h = 0
    hwidth = highlight_width
    [axs_x, axs_y, axs_w, axs_h] = [xwidth, top_buffer, hwidth * h, height]

    # heatmap axes
    hstart = xwidth + (h * hwidth)
    [ax1_x, ax1_y, ax1_w, ax1_h] = [hstart, top_buffer,
                                    cbar_pad-hstart, height]

    # plot dendrogram
    ax_dendrogram = fig.add_axes([axm_x, axm_y, axm_w, axm_h],
                                 frame_on=True)
    _plot_dendrogram(ax_dendrogram, table, edge_list, linewidth=linewidth)

    # plot highlights for dendrogram
    if highlights is not None:
        ax_highlights = fig.add_axes([axs_x, axs_y, axs_w, axs_h],
                                     frame_on=True, sharey=ax_dendrogram)
        _plot_highlights_dendrogram(ax_highlights, table, t, highlights)

    # plot heatmap
    ax_heatmap = fig.add_axes([ax1_x, ax1_y, ax1_w, ax1_h], frame_on=True,
                              sharey=ax_dendrogram)
    cbar = _plot_heatmap(ax_heatmap, table, mdvar,
                         grid_col, grid_width, cmap, **kwargs)
    # plot the colorbar
    fig.colorbar(cbar)
    return fig


# TODO: Refactor and place in utils.  This can be also
# be used for the balance_basis calculations
def _tree_coordinates(t):
    """ Builds a matrix to link tree positions to matrix"""
    # first traverse the tree to count the children
    for n in t.postorder(include_self=True):
        if n.is_tip():
            n._n_tips = 1
        else:
            n._n_tips = sum(c._n_tips for c in n.children)

    for i, n in enumerate(t.levelorder(include_self=True)):
        if n.is_root():
            n._k = 0
            n._t = 0
        else:
            if n is n.parent.children[DENOMINATOR]:
                n._k = n.parent._k + n.parent._r
                n._t = n.parent._t
            else:
                n._k = n.parent._k
                n._t = n.parent._t + n.parent._l

        if n.is_tip():
            continue
        n._r = n.children[NUMERATOR]._n_tips
        n._l = n.children[DENOMINATOR]._n_tips
    return t


def _plot_highlights_dendrogram(ax_highlights, table, t, highlights):
    """ Plots highlights for subtrees in the dendrograms.

    Note that this assumes that the dendrograms are strictly bifurcating
    and the highlights only specify the children for a given subtree.
    """
    offset = 0.5

    num_h = len(highlights)
    hcoords = []
    for i, n in enumerate(highlights.index):
        node = t.find(n)

        k, l, r = node._k, node._l, node._r

        ax_highlights.add_patch(
            patches.Rectangle(
                (i/num_h, k-offset),  # x, y
                1/num_h,  # width
                r,  # height
                facecolor=highlights.iloc[i, 0]
            ))

        ax_highlights.add_patch(
            patches.Rectangle(
                (i/num_h, k+r-offset),  # x, y
                1/num_h,  # width
                l,  # height
                facecolor=highlights.iloc[i, 1]
            ))
        hcoords.append((i+offset)/num_h)
    ax_highlights.set_ylim([-offset, table.shape[0]-offset])
    ax_highlights.set_yticks([])
    ax_highlights.set_xticks(hcoords)
    ax_highlights.set_xticklabels(highlights.index, rotation=90)


def _plot_dendrogram(ax_dendrogram, table, edges, linewidth=1):
    """ Plots the actual dendrogram.

    Parameters
    ----------
    ax_dendrogram : matplotlib axes object
        Contains the matplotlib axes in which the dendrogram will be plotted.
    table : pd.DataFrame
        Contain sample/feature labels along with table of values.
        Rows correspond to samples, and columns correspond to features.
    edges : pd.DataFrame
        (x,y) coordinates for edges in the heatmap.
    """
    offset = 0.5
    for i in range(len(edges.index)):
        row = edges.iloc[i]
        ax_dendrogram.plot([row.x0, row.x1],
                           [row.y0-offset, row.y1-offset], '-k', lw=linewidth)
    ax_dendrogram.set_ylim([-offset, table.shape[0]-offset])
    ax_dendrogram.set_yticks([])
    ax_dendrogram.set_xticks([])
    ax_dendrogram.axis('off')


def _sort_table(table, mdvar):
    """ Sorts metadata category and aligns with table.

    Parameters
    ----------
    table : pd.DataFrame
        Contain sample/feature labels along with table of values.
        Rows correspond to samples, and columns correspond to features.
    mdvar : pd.Series
        Metadata values for samples.  The index must correspond to the
        index of `table`.

    Returns
    -------
    pd.DataFrame
        Aligned feature table.
    pd.Series
        Aligned metadata.
    """
    mdvar = mdvar.sort_values()
    table = table.reindex(columns=mdvar.index)
    return table, mdvar


def _plot_heatmap(ax_heatmap, table, mdvar, grid_col,
                  grid_width, cmap, **kwargs):
    """ Sorts metadata category and aligns with table.

    Parameters
    ----------
    ax_heatmap : matplotlib axes object
        Contains the matplotlib axes in which the heatmap will be plotted.
    table : pd.DataFrame
        Contain sample/feature labels along with table of values.
        Rows correspond to samples, and columns correspond to features.
    mdvar : pd.Series
        Metadata values for samples.  The index must correspond to the
        index of `table`.
    grid_col: str
        Color of vertical lines for highlighting sample metadata.
        (default='w')
    grid_width: int
        Width of vertical lines for highlighting sample metadata.
        (default=2)
    **kwargs: dict
        Arguments to be passed into the heatmap function.

    Returns
    -------
    colorbar_ax : matplotlib axes object
        The colorbar axis.
    """
    # TODO add explicit test for this, since matplotlib orientation
    # is from top to down (i.e. is backwards)
    table, mdvar = _sort_table(table, mdvar)
    table = table.iloc[::-1, :]
    colorbar_ax = ax_heatmap.imshow(table, aspect='auto',
                                    interpolation='nearest',
                                    cmap=cmap, **kwargs)
    ax_heatmap.set_ylim([0, table.shape[0]])
    vcounts = mdvar.value_counts()

    # ensure that the ordering is the same
    vcounts = vcounts.sort_index()

    ticks = vcounts.sort_index().cumsum()
    midpoints = ticks - (ticks - np.array([0] + list(ticks.values[:-1]))) / 2.0
    ax_heatmap.set_xticks(ticks.values-0.5, minor=False)
    ax_heatmap.set_xticklabels([], minor=False)

    ax_heatmap.xaxis.grid(True, which='major', color=grid_col,
                          linestyle='-', linewidth=grid_width)

    ax_heatmap.set_xticks(midpoints-0.5, minor=True)
    ax_heatmap.set_xticklabels(vcounts.index, minor=True)
    ax_heatmap.set_xlabel(mdvar.name)
    return colorbar_ax
