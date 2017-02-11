# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the GPLv3 License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------
import pandas as pd
from gneiss.plot._dendrogram import UnrootedDendrogram

try:
    from bokeh.models.glyphs import Circle, Segment
    from bokeh.models import ColumnDataSource, Range1d, DataRange1d, Plot
    from bokeh.models import HoverTool, BoxZoomTool, ResetTool
except ImportError:
    raise ImportWarning('Bokeh not installed. '
                        '`radialplot` will not be available')


def radialplot(tree, node_hue='node_hue', node_size='node_size',
               node_alpha='node_alpha', edge_hue='edge_hue',
               edge_alpha='edge_alpha', edge_width='edge_width',
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
    default_node_hue = '#D3D3D3'
    nodes[node_hue] = pd.Series({n.name:getattr(n, node_hue,
                                                default_node_hue)
                               for n in t.levelorder(include_self=True)})

    default_node_size = 1
    nodes[node_size] = pd.Series({n.name:getattr(n, node_size,
                                                 default_node_size)
                                for n in t.levelorder(include_self=True)})

    default_node_alpha = 1
    nodes[node_alpha] = pd.Series({n.name:getattr(n, node_alpha,
                                                  default_node_alpha)
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
    ns = [n.name for n in t.levelorder(include_self=True)]
    attrs = pd.DataFrame(index=ns)

    default_edge_hue = '#000000'
    attrs[edge_hue] = pd.Series({n.name:getattr(n, edge_hue,
                                                default_edge_hue)
                                 for n in t.levelorder(include_self=True)})

    default_edge_width = 1
    attrs[edge_width] = pd.Series({n.name:getattr(n, edge_width,
                                                  default_edge_width)
                                  for n in t.levelorder(include_self=True)})


    default_edge_alpha = 1
    attrs[edge_alpha] = pd.Series({n.name:getattr(n, edge_alpha,
                                                  default_edge_alpha)
                                   for n in t.levelorder(include_self=True)})


    edges = pd.merge(edges, attrs, left_on='dest_node',
                     right_index=True, how='outer')


    node_glyph = Circle(x="x", y="y",
                        radius=node_size,
                        fill_color=node_hue,
                        fill_alpha=node_alpha)

    edge_glyph = Segment(x0="x0", y0="y0",
                         x1="x1", y1="y1",
                         line_color=edge_hue,
                         line_alpha=edge_alpha,
                         line_width=edge_width)

    def df2ds(df):
        return ColumnDataSource(ColumnDataSource.from_df(df))

    ydr = DataRange1d(range_padding=0.05)
    xdr = DataRange1d(range_padding=0.05)

    plot = Plot(x_range=xdr, y_range=ydr, **kwargs)
    plot.add_glyph(df2ds(edges), edge_glyph)
    ns = plot.add_glyph(df2ds(nodes), node_glyph)

    # TODO: Will need to make the hovertool options more configurable
    tooltip = """
        <div>
            <span style="font-size: 17px; font-weight: bold;">name: </span>
            <span style="font-size: 15px; color: #151515;">@index</span>
        </div>
    """

    hover = HoverTool(renderers = [ns], tooltips=tooltip)
    plot.add_tools(hover, BoxZoomTool(), ResetTool())

    return plot
