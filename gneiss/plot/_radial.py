# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------
import pandas as pd
from gneiss.plot._dendrogram import UnrootedDendrogram
from bokeh.models.glyphs import Circle, Segment
from bokeh.models import ColumnDataSource, DataRange1d, Plot
from bokeh.models import (HoverTool, BoxZoomTool, ResetTool,
                          WheelZoomTool, SaveTool, PanTool)


def radialplot(tree, node_color='node_color', node_size='node_size',
               node_alpha='node_alpha', edge_color='edge_color',
               edge_alpha='edge_alpha', edge_width='edge_width',
               hover_var='hover_var', figsize=(500, 500), **kwargs):
    """ Plots unrooted radial tree.

    Parameters
    ----------
    tree : instance of skbio.TreeNode
       Input tree for plotting.
    node_color : str
       Name of variable in `tree` to color nodes.
    node_size : str
       Name of variable in `tree` that specifies the radius of nodes.
    node_alpha : str
       Name of variable in `tree` to specify node transparency.
    edge_color : str
       Name of variable in `tree` to color edges.
    edge_alpha : str
       Name of variable in `tree` to specify edge transparency.
    edge_width : str
       Name of variable in `tree` to specify edge width.
    hover_var : str
       Name of variable in `tree` to display in the hover menu.
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
    # TODO: Add in example doc string

    # This entire function was motivated by
    # http://chuckpr.github.io/blog/trees2.html
    t = UnrootedDendrogram.from_tree(tree.copy())

    nodes = t.coords(figsize[0], figsize[1])

    # fill in all of the node attributes
    def _retreive(tree, x, default):
        return pd.Series({n.name: getattr(n, x, default)
                          for n in tree.levelorder()})

    # default node color to light grey
    nodes[node_color] = _retreive(t, node_color, default='#D3D3D3')
    nodes[node_size] = _retreive(t, node_size, default=1)
    nodes[node_alpha] = _retreive(t, node_alpha, default=1)
    nodes[hover_var] = _retreive(t, hover_var, default=None)

    edges = nodes[['child0', 'child1']]
    edges = edges.dropna(subset=['child0', 'child1'])
    edges = edges.unstack()
    edges = pd.DataFrame({'src_node': edges.index.get_level_values(1),
                          'dest_node': edges.values})
    edges['x0'] = [nodes.loc[n].x for n in edges.src_node]
    edges['x1'] = [nodes.loc[n].x for n in edges.dest_node]
    edges['y0'] = [nodes.loc[n].y for n in edges.src_node]
    edges['y1'] = [nodes.loc[n].y for n in edges.dest_node]
    ns = [n.name for n in t.levelorder(include_self=True)]
    attrs = pd.DataFrame(index=ns)

    # default edge color to black
    attrs[edge_color] = _retreive(t, edge_color, default='#000000')
    attrs[edge_width] = _retreive(t, edge_width, default=1)
    attrs[edge_alpha] = _retreive(t, edge_alpha, default=1)

    edges = pd.merge(edges, attrs, left_on='dest_node',
                     right_index=True, how='outer')
    edges = edges.dropna(subset=['src_node'])

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
    ns = plot.add_glyph(df2ds(nodes), node_glyph)

    tooltip = [
        ("Feature ID", "@index")
    ]
    if hover_var is not None:
        tooltip += [(hover_var, "@" + hover_var)]

    hover = HoverTool(renderers=[ns], tooltips=tooltip)
    plot.add_tools(hover, BoxZoomTool(), ResetTool(),
                   WheelZoomTool(), SaveTool(), PanTool())

    return plot
