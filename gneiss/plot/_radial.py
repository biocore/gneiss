# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the GPLv3 License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------
import pandas as pd

try:
    from bokeh.io import output_notebook, show
    from bokeh.models.glyphs import Circle, Segment
    from bokeh.models import ColumnDataSource, Range1d, DataRange1d, Plot
except ImportError:
    raise ImportWarning('Bokeh not installed. '
                        '`radialplot` will not be available')


def radialplot(tree, node_hue, node_size, node_alpha
               edge_hue, edge_alpha, edge_width,
               figsize=(500, 500), **kwargs):
    """ Plots unrooted radial tree.

    Parameters
    ----------
    tree : instance of skbio.TreeNode
       Input tree for plotting.
    node_hue : str
       Name of variable in `tree` to color nodes.
    node_size : str
       Name of variable in `tree` that species the radius of nodes.
    node_alpha : str
       Name of variable in `tree` to specify node transparency.
    edge_hue : str
       Name of variable in `tree` to color edges.
    edge_alpha : str
       Name of variable in `tree` to specify edge transparency.
    edge_width : str
       Name of variable in `tree` to specify edge width.
    figsize : tuple, int
       Size of resulting figure.  default: (500, 500)
    **kwargs: dict
       Plotting options to pass into bokeh.models.Plot

    Returns
    -------
    bokeh.models.Plot
       Interactive plotting instance.

    Notes
    -----
    This assumes that the tree is strictly bifurcating.

    See also
    --------
    bifurcate
    """
    # This entire function was motivated by
    # http://chuckpr.github.io/blog/trees2.html
    t = UnrootedDendrogram.from_tree(tree)

    nodes = t.coords(figsize[0], figsize[1])

    # fill in all of the node attributes
    nodes[node_hue] = pd.Series({n.name:getattr(n, node_hue)
                               for n in t.levelorder(include_self=True)})
    nodes[node_size] = pd.Series({n.name:getattr(n, node_size)
                                for n in t.levelorder(include_self=True)})
    nodes[node_alpha] = pd.Series({n.name:getattr(n, node_alpha)
                                 for n in t.levelorder(include_self=True)})

    edges = nodes[['child0', 'child1']]
    edges = edges.dropna(subset=['child0', 'child1'])
    edges = edges.unstack()
    edges = pd.DataFrame({'src_node': edges.index.get_level_values(1),
                          'dest_node' : edges.values})
    edges['x0'] = [nodes.loc[n].x for n in edges.src_node]
    edges['x1'] = [nodes.loc[n].x for n in edges.dest_node]
    edges['y0'] = [nodes.loc[n].y for n in edges.src_node]
    edges['y1'] = [nodes.loc[n].y for n in edges.dest_node]

    edges[edge_hue] = pd.Series({n.name:getattr(n, edge_hue)
                                 for n in t.levelorder(include_self=True)})
    edges[edge_size] = pd.Series({n.name:getattr(n, edge_size)
                                  for n in t.levelorder(include_self=True)})
    edges[edge_alpha] = pd.Series({n.name:getattr(n, edge_alpha)
                                   for n in t.levelorder(include_self=True)})

    node_glyph = Circle(x="x", y="y",
                        radius=node_size,
                        fill_color=node_color,
                        fill_alpha=node_alpha)

    edge_glyph = Segment(x0="x0", y0="y0",
                         x1="x1", y1="y1",
                         line_color=edge_color,
                         line_alpha=edge_alpha,
                         line_width=edge_width)

    def df2ds(df):
        return ColumnDataSource(ColumnDataSource.from_df(df))

    ydr = DataRange1d(range_padding=0.05)
    xdr = DataRange1d(range_padding=0.05)

    plot = Plot(x_range=xdr, y_range=ydr, **kwargs)
    plot.add_glyph(df2ds(edges), edge_glyph)
    plot.add_glyph(df2ds(nodes), node_glyph)

    return plot
