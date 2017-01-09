# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the GPLv3 License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------
import ete3
from ete3 import TreeStyle, AttrFace, NodeStyle
from ete3.treeview import faces
from PyQt4.QtGui import (QGraphicsPolygonItem,
                         QPen, QColor, QBrush, QPolygonF)
from PyQt4.QtCore import QPointF
from ete3.treeview.faces import StaticItemFace, Face
import pandas as pd


class _DiamondItem(QGraphicsPolygonItem):
    def __init__(self, width, height, label, color='#0000FF'):

        self.pol = QPolygonF()

        self.pol = QPolygonF()
        self.pol.append(QPointF(width / 2.0, 0))
        self.pol.append(QPointF(width, height / 2.0))
        self.pol.append(QPointF(width / 2.0, height))
        self.pol.append(QPointF(0, height / 2.0))
        self.pol.append(QPointF(width / 2.0, 0))

        self.label = label
        QGraphicsPolygonItem.__init__(self, self.pol)

        self.setBrush(QBrush(QColor(color)))
        self.setPen(QPen(QColor(color)))

    def paint(self, p, option, widget):
        super(_DiamondItem, self).paint(p, option, widget)
        ete3.treeview.faces._label_painter(self, p, option, widget)


class CollapsedDiamondFace(StaticItemFace, Face):
    """
    Creates a collapsed node face object.

    """
    def __init__(self, width, height, label='', color='#0000FF'):
        Face.__init__(self)
        self.height = height
        self.width = width
        self.type = 'item'
        self.label = label
        self.color = color

    def update_items(self):
        self.item = _DiamondItem(width=self.width, height=self.height,
                                 label=self.label, color=self.color)

    def _width(self):
        return self.width

    def _height(self):
        return self.height


def diamondtree(tree, collapsers=None, **kwargs):
    """ Plots collapsed tree with background coloring and clade coloring.

    This creates collapsed trees similar to the tree plots in the tree of
    life paper [1]. Rather than plotting all of the leaves, specified
    subtrees will be replaced by diamonds that are scaled to approximate
    the depth and the width of the subtree.

    Parameters
    ----------
    tree : skbio.TreeNode
        A strictly bifurcating tree defining a hierarchical relationship
        between all of the features within `table`.
    collapsers : list of str
        Names of internal nodes to collapse within the tree.
    layout : function, optional
        A layout for formatting the tree visualization. Must take a
        `ete.tree` as a parameter.
    labelcolor: str
        Color of the node labels. (default 'black')
    bgcolors: dict of str or matplotlib colormap
        String or function encoding matplotlib colormap for the backgrounds
        outside of the clades.
    cladecolors: dict of str or str
        String or function encoding matplotlib colormap for the colors
        within the clade faces. (default '#0000FF')
    depth_scaling : int
        Scaling factor for height of the subtrees represented by diamonds.
        (default : 20)
    breadth_scaling : int
        Scaling factor for width of the subtrees represented by diamonds.
        (default : 10)
    label_size : int
        Size of nodes labels.
    mode : str
        Type of display to show the tree. ('c': circular, 'r': rectangular).


    Returns
    -------
    ete.Tree
        ETE tree object that will be plotted.
    ete.TreeStyle
        ETE TreeStyle that decorates the tree and heatmap visualization.

    References
    ----------
    .. [1] Hug, Laura A., et al. "A new view of the tree of life."
       Nature Microbiology 1 (2016): 16048.
    """

    # TODO: Allow for the option to encode labels in different colors
    # (i.e. pass in a pandas series)
    params = {'bgcolors': None, 'cladecolors': '#0000FF',
              'labelcolor': 'black', 'label_size': 10,
              'depth_scaling': 30, 'breadth_scaling': 6,
              'mode': 'c',
              # TODO: Enable layout
              # layout : function, optional
              #    A layout for formatting the tree visualization. Must take a
              #    `ete.tree` as a parameter.
              # For now just define a null function if no layout is defined
              # TODO: Learning scaling factors for depth and breadth
              'layout': lambda x: x}
    for key in params:
        params[key] = kwargs.get(key, params[key])

    bgcolors = params['bgcolors']
    cladecolors = params['cladecolors']
    labelcolor = params['labelcolor']
    label_size = params['label_size']
    depth_scaling = params['depth_scaling']
    breadth_scaling = params['breadth_scaling']
    mode = params['mode']
    layout = params['layout']

    tr = ete3.Tree.from_skbio(tree)

    def diamond_layout(node):
        # Run the layout passed in first before
        # filling in the heatmap
        layout(node)

        N = AttrFace("name", fsize=label_size, fgcolor=labelcolor)

        # background colors
        c, found = _get_node_color(bgcolors, node, "")
        if found:
            nst = NodeStyle()
            nst["bgcolor"] = c
            node.set_style(nst)
        if node.name in collapsers:
            # scaling factor for approximating subtree depth
            w = node.get_farthest_node()[1]*depth_scaling
            # scaling factor for approximating for subtree width
            h = len(node)*breadth_scaling
            c, _ = _get_node_color(cladecolors, node, "#0000FF")

            C = CollapsedDiamondFace(width=w, height=h, color=c)
            node.img_style['draw_descendants'] = False

            # And place as a float face over the tree
            faces.add_face_to_node(C, node, 0, position="float")
            faces.add_face_to_node(N, node, 1, position="float")
        else:
            faces.add_face_to_node(N, node, 0)

    ts = TreeStyle()

    # Draw a tree
    ts.mode = mode

    # We will add node names manually
    ts.show_leaf_name = False
    # Show branch data
    ts.show_branch_length = True
    ts.show_branch_support = True

    ts.layout_fn = diamond_layout
    return tr, ts


def _get_node_color(x, node, default_color):
    """ Retrieves color from dict, Series, str.

    Parameters
    ----------
    x : dict or pd.Series or str
        Input color(s).
    node : ete.TreeNode
        Input tree.
    default_color: str
        The default color to set to if the color is not present in `x`

    Returns
    -------
    str :
       The color for `node`
    bool :
       Indicates if the node color was found in `x`
    """
    try:
        if isinstance(x, str):
            return x, True
        elif isinstance(x, pd.Series):
            return x.loc[node.name], True
        elif isinstance(x, dict):
            return x[node.name], True
        else:
            raise TypeError("color type %s not supported." % type(x).__name__)
    except KeyError:
        # Use default if the color isn't specified
        c = default_color
        return c, False
