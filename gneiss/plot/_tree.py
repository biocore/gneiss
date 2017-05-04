# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------
import pandas as pd
from gneiss.plot._dendrogram import SquareDendrogram
import matplotlib.pyplot as plt


def dendrogram(tree, labels=True, linecolor='k', textcolor='k',
               figsize=(5, 5), lineargs=dict(), textargs=dict()):
    """ Plots a dendrogram.

    Parameters
    ----------
    tree: skbio.TreeNode
        Tree to be plotted.
    labels: bool
        Specifies if the labels will be plotted.  If true, then
        the `name` attribute under each node in the tree will be
        rendered as labels.
    linecolor: str
        Name of color for plotting the dendrogram lines.
    textcolor: str
        Name of color for plotting the text.
    figsize: tuple of int
        Specifies (width, height) for figure. (default=(5, 5))
    lineargs: dict
        Plotting properties for the dendrogram lines.
    textargs: dict
        Plotting properties for the text.

    Returns
    -------
    matplotlib.pyplot.figure
        Matplotlib figure object

    """
    t = SquareDendrogram.from_tree(tree.copy())

    pts = t.coords(width=figsize[0], height=figsize[1])


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
            {'x0': sx, 'y0': sy, 'x1': sx, 'y1': dy, 'name': src}
        )
        edge_list.append(
            {'x0': sx, 'y0': dy, 'x1': dx, 'y1': dy, 'name': ''}
        )

    edges = pd.DataFrame(edge_list)
    fig, ax_dendrogram = plt.subplots(figsize=figsize, facecolor='white')

    offset = 0.5
    for i in range(len(edges.index)):
        row = edges.iloc[i]
        ax_dendrogram.plot([row.x0, row.x1], [row.y0-offset, row.y1-offset],
                           color=linecolor, **lineargs)

    for i in range(len(pts.index)):
        row = pts.iloc[i]
        ax_dendrogram.text(row.x, row.y-offset, pts.iloc[i].name,
                           color=textcolor, **textargs)

    ax_dendrogram.set_yticks([])
    ax_dendrogram.set_xticks([])
    ax_dendrogram.axis('off')
    return fig
