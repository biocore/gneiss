"""
Plotting functions (:mod:`gneiss.plot`)
===============================================

.. currentmodule:: gneiss.plot

This module contains plotting functionality

Functions
---------

.. autosummary::
   :toctree: generated/

   heatmap

"""

import numpy as np
from ete3 import TreeStyle, AttrFace, ProfileFace
from ete3 import ClusterNode
from ete3.treeview.faces import add_face_to_node
import io


def heatmap(table, tree, cmap='viridis', **kwargs):
    """ Plots tree on heatmap

    Parameters
    ----------
    table : pd.DataFrame
        Contingency table where samples correspond to rows and
        features correspond to columns.
    tree : skbio.TreeNode
        A strictly bifurcating tree defining a hierarchical relationship
        between all of the features within `table`.
    cmap: matplotlib colormap
        String or function encoding matplotlib colormap.
    labelcolor: str
        Color of the node labels. (default 'black')
    rowlabel_size : int
        Size of row labels. (default 8)
    width : int
        Heatmap cell width. (default 200)
    height : int
        Heatmap cell height (default 14)

    Returns
    -------
    ete.Tree
        ETE tree object that will be plotted.
    ete.TreeStyle
        ETE TreeStyle that decorates the tree and heatmap visualization.
    """
    # TODO: Allow for the option to encode labels in different colors
    # (i.e. pass in a pandas series)
    params = {'rowlabel_size': 8, 'width': 200, 'height': 14,
              'cmap': 'viridis', 'labelcolor': 'black',
              # TODO: Enable layout
              # layout : function, optional
              #    A layout for formatting the tree visualization. Must take a
              #    `ete.tree` as a parameter.
              'layout': lambda x: x}

    for key in params.keys():
        params[key] = kwargs.get(key, params[key])
    fsize = params['rowlabel_size']
    width = params['width']
    height = params['height']
    colorscheme = params['cmap']
    layout = params['layout']

    # Allow for matplotlib colors to be encoded in ETE3 heatmap
    # Originally from https://github.com/lthiberiol/virfac
    def get_color_gradient(self):
        from PyQt4 import QtGui
        try:
            import matplotlib.pyplot as plt
            import matplotlib.colors as colors
            import matplotlib.cm as cmx
        except:
            ImportError("Matplotlib not installed.")

        cNorm = colors.Normalize(vmin=0, vmax=1)
        scalarMap = cmx.ScalarMappable(norm=cNorm,
                                       cmap=plt.get_cmap(self.colorscheme))
        color_scale = []
        for scale in np.linspace(0, 1, 255):
            [r, g, b, a] = scalarMap.to_rgba(scale, bytes=True)
            color_scale.append(QtGui.QColor(r, g, b, a))
        return color_scale

    ProfileFace.get_color_gradient = get_color_gradient
    tree.name = ""

    f = io.StringIO()
    table.T.to_csv(f, sep='\t', index_label='#Names')

    tr = ClusterNode(str(tree), text_array=str(f.getvalue()))
    matrix_max = np.max(table.values)
    matrix_min = np.min(table.values)
    matrix_avg = matrix_min + ((matrix_max - matrix_min) / 2)

    # Encode the actual profile face
    nameFace = AttrFace("name", fsize=fsize)

    def heatmap_layout(node):
        # Run the layout passed in first before
        # filling in the heatmap
        layout(node)

        if node.is_leaf():
            profileFace = ProfileFace(
                values_vector=table.loc[:, node.name].values,
                style="heatmap",
                max_v=matrix_max, min_v=matrix_min,
                center_v=matrix_avg,
                colorscheme=colorscheme,
                width=width, height=height)

            add_face_to_node(profileFace, node, 0, aligned=True)
            node.img_style["size"] = 0
            add_face_to_node(nameFace, node, 1, aligned=True)

    ts = TreeStyle()
    ts.layout_fn = heatmap_layout

    return tr, ts
