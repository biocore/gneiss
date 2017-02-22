# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the GPLv3 License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------
from skbio import TreeNode
import pandas as pd
import numpy
import abc


class Dendrogram(TreeNode):
    """ Stores data to be plotted as a dendrogram.

    A `Dendrogram` object is represents a tree in addition to the
    key information required to create a tree layout prior to
    visualization.  No layouts are specified within this class,
    since this serves as a super class for different tree layouts.

    Parameters
    ----------
    use_lengths: bool
        Specifies if the branch lengths should be included in the
        resulting visualization (default True).

    Attributes
    ----------
    length
    """
    aspect_distorts_lengths = True

    def __init__(self, use_lengths=True, **kwargs):
        """ Constructs a Dendrogram object for visualization.

        """
        super().__init__(**kwargs)
        self.use_lengths_default = use_lengths

    def _cache_ntips(self):
        for n in self.postorder():
            if n.is_tip():
                n._n_tips = 1
            else:
                n._n_tips = sum(c._n_tips for c in n.children)

    def coords(self, height, width):
        """ Returns coordinates of nodes to be rendered in plot.

        Parameters
        ----------
        height : int
            The height of the canvas.
        width : int
            The width of the canvas.

        Returns
        -------
        pd.DataFrame
            index : str
                Name of node.
            x : float
                x-coordinate of node.
            y : float
                y-coordinate of node.
            child(i) : str
                Name of ith child node in that specific node.
                in the tree.
            is_tip : str
                Specifies if the node is a tip in the treee.
        """
        self.rescale(width, height)
        result = {}
        for node in self.postorder():
            children = {'child%d' % i: n.name
                        for i, n in enumerate(node.children)}
            coords = {'x': node.x2, 'y': node.y2}
            is_tip = {'is_tip': node.is_tip()}
            result[node.name] = {**coords, **children, **is_tip}
        result = pd.DataFrame(result).T

        # reorder so that x and y are first
        cols = ['x', 'y'] + sorted(list(set(result.columns) - set(['x', 'y'])))
        return result.loc[:, cols]

    @abc.abstractmethod
    def rescale(self, width, height):
        pass


class UnrootedDendrogram(Dendrogram):
    """ Stores data to be plotted as an unrooted dendrogram.

    A `UnrootedDendrogram` object is represents a tree in addition to the
    key information required to create a radial tree layout prior to
    visualization.

    Parameters
    ----------
    use_lengths: bool
        Specifies if the branch lengths should be included in the
        resulting visualization (default True).

    Attributes
    ----------
    length
    """
    aspect_distorts_lengths = True

    def __init__(self, **kwargs):
        """ Constructs a UnrootedDendrogram object for visualization.

        Parameters
        ----------
        use_lengths: bool
            Specifies if the branch lengths should be included in the
            resulting visualization (default True).
        """
        super().__init__(**kwargs)

    @classmethod
    def from_tree(cls, tree):
        """ Creates an UnrootedDendrogram object from a skbio tree.

        Parameters
        ----------
        tree : skbio.TreeNode
            Input skbio tree

        Returns
        -------
        UnrootedDendrogram
        """
        for n in tree.postorder():
            n.__class__ = UnrootedDendrogram
        tree._cache_ntips()
        return tree

    def rescale(self, width, height):
        """ Find best scaling factor for fitting the tree in the dimensions
        specified by width and height.

        This method will find the best orientation and scaling possible
        to fit the tree within the dimensions specified by width and height.

        Parameters
        ----------
        width : float
            width of the canvas
        height : float
            height of the canvas

        Returns
        -------
        best_scaling : float
            largest scaling factor in which the tree can fit in the canvas.

        Notes
        -----
        """
        angle = (2 * numpy.pi) / self._n_tips
        # this loop is a horrible brute force hack
        # there are better (but complex) ways to find
        # the best rotation of the tree to fit the display.
        best_scale = 0
        for i in range(60):
            direction = i / 60.0 * numpy.pi
            # TODO:
            # This function has a little bit of recursion.  This will
            # need to be refactored to remove the recursion.

            points = self.update_coordinates(1.0, 0, 0, direction, angle)
            xs, ys = zip(*points)
            # double check that the tree fits within the margins
            scale = min(float(width) / (max(xs) - min(xs)),
                        float(height) / (max(ys) - min(ys)))
            # TODO: This margin seems a bit arbituary.
            # will need to investigate.
            scale *= 0.95  # extra margin for labels
            if scale > best_scale:
                best_scale = scale
                mid_x = width / 2 - ((max(xs) + min(xs)) / 2) * scale
                mid_y = height / 2 - ((max(ys) + min(ys)) / 2) * scale
                best_args = (scale, mid_x, mid_y, direction, angle)

        self.update_coordinates(*best_args)
        return best_scale

    def update_coordinates(self, s, x1, y1, a, da):
        """ Update x, y coordinates of tree nodes in canvas.

        `update_coordinates` will recursively updating the
        plotting parameters for all of the nodes within the tree.
        This can be applied when the tree becomes modified (i.e. pruning
        or collapsing) and the resulting coordinates need to be modified
        to reflect the changes to the tree structure.

        Parameters
        ----------
        s : float
            scaling
        x1 : float
            x midpoint
        y1 : float
            y midpoint
        a : float
            angle (degrees)
        da : float
            angle resolution (degrees)

        Returns
        -------
        points : list of tuple
            2D coordinates of all of the nodes.

        Notes
        -----
        This function has a little bit of recursion.  This will
        need to be refactored to remove the recursion.
        """
        # Constant angle algorithm.  Should add maximum daylight step.
        x2 = x1 + self.length * s * numpy.sin(a)
        y2 = y1 + self.length * s * numpy.cos(a)
        (self.x1, self.y1, self.x2, self.y2, self.angle) = (x1, y1, x2, y2, a)
        # TODO: Add functionality that allows for collapsing of nodes
        a = a - self._n_tips * da / 2
        if self.is_tip():
            points = [(x2, y2)]
        else:
            points = []
            # TODO:
            # This function has a little bit of recursion.  This will
            # need to be refactored to remove the recursion.
            for child in self.children:
                # calculate the arc that covers the subtree.
                ca = child._n_tips * da
                points += child.update_coordinates(s, x2, y2, a+ca/2, da)
                a += ca
        return points
