# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the GPLv3 License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

"""Drawing trees.

Draws horizontal trees where the vertical spacing between taxa is
constant.  Since dy is fixed dendrograms can be either:
 - square:     dx = distance
 - not square: dx = max(0, sqrt(distance**2 - dy**2))

Also draws basic unrooted trees.

For drawing trees use either:
 - UnrootedDendrogram

Note: This is directly ported from pycogent.
"""

# Future:
#  - font styles
#  - orientation switch
# Layout gets more complicated for rooted tree styles if dy is allowed to vary,
# and constant-y is suitable for placing alongside a sequence alignment anyway.
from skbio import TreeNode
import pandas as pd
import numpy
import abc


class Dendrogram(TreeNode):
    # One of these for each tree edge.  Extra attributes:
    #    depth - distance from root to bottom of edge
    #    height - max distance from a decendant leaf to top of edge
    #    width - number of decendant leaves
    # note these are named tree-wise, not geometricaly, so think
    # of a vertical tree (for this part anyway)
    #
    #   x1, y1, x2, y2 - coordinates
    # these are horizontal / vertical as you would expect
    #
    # The algorithm is split into 4 passes over the tree for easier
    # code reuse - vertical drawing, new tree styles, new graphics
    # libraries etc.

    aspect_distorts_lengths = True

    def __init__(self, use_lengths=True, **kwargs):
        """ Constructs a Dendrogram object for visualization.

        Parameters
        ----------
        use_lengths: bool
            Specifies if the branch lengths should be included in the
            resulting visualization (default True).
        """
        super().__init__(**kwargs)
        self.use_lengths_default = use_lengths

    def _cache_ntips(self):
        for n in self.postorder(include_self=True):
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
                name of node
            x : float
                x-coordinate of point
            y : float
                y-coordinate of point
            left : str
                name of left child node
            right : str
                name of right child node
        """
        self.rescale(width, height)
        names = [n.name for n in self.postorder(include_self=True)]
        result = {}
        for node in self.postorder(include_self=True):
            children = {'child%d' % i : n.name
                        for i, n in enumerate(node.children)}
            coords = {'x': node.x2, 'y': node.y2}
            is_tip = {'is_tip': node.is_tip()}
            result[node.name] = {**coords, **children, **is_tip}
        result = pd.DataFrame(result).T

        # reorder so that x and y are first
        cols = list(result)
        cols.insert(0, cols.pop(cols.index('y')))
        cols.insert(0, cols.pop(cols.index('x')))
        result = result.ix[:, cols]
        return result

    @abc.abstractmethod
    def rescale(self, width, height):
        pass

class UnrootedDendrogram(Dendrogram):
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
        for n in tree.postorder(include_self=True):
            n.__class__ = UnrootedDendrogram
        tree._cache_ntips()
        return tree

    def rescale(self, width, height):
        """ Find best scaling factor for fitting the tree in the canvas

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
        This function has a little bit of recursion.  This will
        need to be refactored to remove the recursion.
        """

        angle = 2*numpy.pi / self._n_tips
        # this loop is a horrible brute force hack
        # there are better (but complex) ways to find
        # the best rotation of the tree to fit the display.
        best_scale = 0
        for i in range(60):
            direction = i / 60.0 * numpy.pi
            points = self.update_coordinates(1.0, 0, 0, direction, angle)
            xs = [x for (x, y) in points]
            ys = [y for (x, y) in points]

            # double check that the tree fits within the margins
            scale = min(float(width) / (max(xs) - min(xs)),
                        float(height) / (max(ys) - min(ys)))
            scale *= 0.95 # extra margin for labels
            if scale > best_scale:
                best_scale = scale
                mid_x = width / 2 - ((max(xs) + min(xs)) / 2) * scale
                mid_y = height / 2 - ((max(ys) + min(ys)) / 2) * scale
                best_args = (scale, mid_x, mid_y, direction, angle)

        self.update_coordinates(*best_args)
        return best_scale

    def update_coordinates(self, s, x1, y1, a, da):
        """ Update x, y coordinates of tree nodes in canvas.

        Parameters
        ----------
        s : float
            scaling
        x1 : float
            x midpoint
        y1 : float
            y midpoint
        a : float
            direction (degrees)
        da : float
            angle (degrees)

        Returns
        -------
        points : list of tuple
            2D coordinates of all of the notes

        Notes
        -----
        This function has a little bit of recursion.  This will
        need to be refactored to remove the recursion.
        """
        # Constant angle algorithm.  Should add maximum daylight step.
        (x2, y2) = (x1+self.length*s*numpy.sin(a), y1+self.length*s*numpy.cos(a))
        (self.x1, self.y1, self.x2, self.y2, self.angle) = (x1, y1, x2, y2, a)
        p = self.parent.name if self.parent is not None else 'None'

        # TODO: Add functionality that allows for collapsing of nodes
        a = a - self._n_tips * da / 2
        if self.is_tip():
            points = [(x2, y2)]
        else:
            points = []
            # recurse down the tree
            for child in self.children:
                ca = child._n_tips * da
                points += child.update_coordinates(s, x2, y2, a+ca/2, da)
                a += ca
        return points
