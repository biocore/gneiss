# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the GPLv3 License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------
import numpy as np
import pandas as pd
from skbio.stats.composition import clr_inv
from collections import OrderedDict
from ete3 import Tree, TreeStyle, AttrFace, ProfileFace
from ete3 import ClusterNode
from ete3.treeview.faces import add_face_to_node
from gneiss.layouts import default_layout
from gneiss.balances import _attach_balances
import io

from PyQt4.QtGui import (QGraphicsRectItem, QGraphicsLineItem,
                         QGraphicsPolygonItem, QGraphicsEllipseItem,
                         QPen, QColor, QBrush, QPolygonF, QFont,
                         QPixmap, QFontMetrics, QPainter,
                         QRadialGradient, QGraphicsSimpleTextItem, QGraphicsTextItem,
                         QGraphicsItem)
from PyQt4.QtCore import Qt,  QPointF, QRect, QRectF
from ete3.treeview.faces import StaticItemFace, Face


class QGraphicsBlobItem(QGraphicsPolygonItem):
    def __init__(self, points):
        """
        Creates a blob from a list of points.

        Parameters
        ----------
        points : list of tuple
             List of points with (x, y) coordinates.
        """
        self.tri = QPolygonF()
        for p in points:
            self.tri.append(QPointF(*p))
        QGraphicsPolygonItem.__init__(self, self.tri)


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
        super(_TriangleItem, self).paint(p, option, widget)
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

    def update_items(self):
        t = _TriangleItem(width=self.width, height=self.height, label=self.label)
        #print(dir(t))
        #c = _SphereItem(10, )
        self.item = t
    def _width(self):
        return self.width

    def _height(self):
        return self.height


def diamondtree(tree, collapsers=None, layout=None,
                bgcolors=None, cladecolors=None, fgcolor='black',
                nodelabel_size=None, leaflabel_size=None,
                depth_scaling=20, breadth_scaling=10,
                **kwargs):
    """ Plots collapsed tree with background coloring and clade coloring.

    This creates collapsed trees similar to the tree plots in the tree of
    life paper [1]. Rather than plotting all of the leaves, specified
    subtrees will be replaced by diamonds that are scaled to approximate
    the depth and the width of the subtree.

    Parameters
    ----------
    table : pd.DataFrame
        Contingency table where samples correspond to rows and
        features correspond to columns.
    collapsers : list of str
        Names of internal nodes to collapse within the tree.
    tree : skbio.TreeNode
        A strictly bifurcating tree defining a hierarchical relationship
        between all of the features within `table`.
    layout : function, optional
        A layout for formatting the tree visualization. Must take a
        `ete.tree` as a parameter.
    bgcolors: dict of str or matplotlib colormap
        String or function encoding matplotlib colormap for the backgrounds
        outside of the clades.
    cladecolors: dict of str or matplotlib colormap
        String or function encoding matplotlib colormap for the colors
        within the clade faces.
    depth_scaling : int
        Scaling factor for height of the subtrees represented by diamonds.
        (default : 20)
    breadth_scaling : int
        Scaling factor for width of the subtrees represented by diamonds.
        (default : 10)
    nodelabel_size : int
        Size of labels of internal nodes.
    leaflabel_size : int
        Size of labels of leaf nodes.

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

    N = AttrFace("name", fsize=fsize, fgcolor=fgcolor)

    def diamond_layout(node):
        if node.name in collapsers:
            # scaling factor for approximating subtree depth
            w = node.get_farthest_node()[1]*depth_scaling
            # scaling factor for approximating for subtree width
            h = len(node)*breadth_scaling
            C = CollapsedDiamondFace(width=w, height=h)
            node.img_style['draw_descendants']=False

            # And place as a float face over the tree
            faces.add_face_to_node(C, node, 0, position="float")
            faces.add_face_to_node(N, node, 1, position="float")
        else:
            faces.add_face_to_node(N, node, 0, position="float")

    ts = TreeStyle()
    ts.layout_fn = diamond_layout

