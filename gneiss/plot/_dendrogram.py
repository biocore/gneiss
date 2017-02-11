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
import abc
from collections import namedtuple
import pandas as pd
import numpy
from skbio import TreeNode


def _sign(x):
    """Returns True if x is positive, False otherwise."""
    return x and x/abs(x)


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
                n.leafcount = 1
            else:
                n.leafcount = sum(c.leafcount for c in n.children)

    def update_geometry(self, use_lengths, depth=None):
        """Calculate tree node attributes such as height and depth.
        Despite the name this first pass is ignorant of issues like
        scale and orientation"""
        if self.length is None or not use_lengths:
            if depth is None:
                self.length = 0
            else:
                self.length = 1
        else:
            self.length = self.length

        self.depth = (depth or 0) + self.length

        children = self.children
        if children:
            for c in children:
                c.update_geometry(use_lengths, self.depth)
            self.height = max([c.height for c in children]) + self.length
            self.leafcount  = sum([c.leafcount for c in children])

        else:
            self.height = self.length
            self.leafcount = self.edgecount = 1

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
        result = {}
        for node in self.postorder(include_self=True):
            children = {'child%d' % i: n.name
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
    def from_tree(cls, tree, use_lengths=True):
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

        tree.update_geometry(use_lengths)
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

        angle = 2*numpy.pi / self.leafcount
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
        x2 = x1+self.length*s*numpy.sin(a)
        y2 = y1+self.length*s*numpy.cos(a)
        (self.x1, self.y1, self.x2, self.y2, self.angle) = (x1, y1, x2, y2, a)
        # TODO: Add functionality that allows for collapsing of nodes
        a = a - self.leafcount * da / 2
        if self.is_tip():
            points = [(x2, y2)]
        else:
            points = []
            # recurse down the tree
            for child in self.children:
                ca = child.leafcount * da
                points += child.update_coordinates(s, x2, y2, a+ca/2, da)
                a += ca
        return points


Dimensions = namedtuple('Dimensions', ['x', 'y', 'height'], verbose=False)

class RootedDendrogram(Dendrogram):
    """RootedDendrogram subclasses provide ycoords and xcoords, which examine
    attributes of a node (its length, coodinates of its children) and return
    a tuple for start/end of the line representing the edge."""

    def width_required(self):
        return self.leafcount

    @abc.abstractmethod
    def xcoords(self, scale, x1):
        pass

    @abc.abstractmethod
    def ycoords(self, scale, y1):
        pass

    def rescale(self, width, height):
        """ Update x, y coordinates of tree nodes in canvas.

        Parameters
        ----------
        scale : Dimensions
            Scaled dimensions of the tree
        x1 : int
            X-coordinate of parent
        """
        xscale = width / self.height
        yscale = height / self.width_required()
        scale = Dimensions(xscale, yscale, self.height)

        # y coords done postorder, x preorder, y first.
        # so it has to be done in 2 passes.
        self.update_y_coordinates(scale)
        self.update_x_coordinates(scale)
        return xscale

    def update_y_coordinates(self, scale, y1=None):
        """The second pass through the tree.  Y coordinates only
        depend on the shape of the tree and yscale.

        Parameters
        ----------
        scale : Dimensions
            Scaled dimensions of the tree
        x1 : int
            X-coordinate of parent
        """
        if y1 is None:
            y1 = self.width_required() * scale.y
        child_y = y1
        for child in self.children:
            child.update_y_coordinates(scale, child_y)
            child_y -= child.width_required() * scale.y
        (self.y1, self.y2) = self.ycoords(scale, y1)

    def update_x_coordinates(self, scale, x1=0):
        """For non 'square' styles the x coordinates will depend
        (a bit) on the y coodinates, so they should be done first.

        Parameters
        ----------
        scale : Dimensions
            Scaled dimensions of the tree
        x1 : int
            X-coordinate of parent
        """
        (self.x1, self.x2) = self.xcoords(scale, x1)
        for child in self.children:
            child.update_x_coordinates(scale, self.x2)


class SquareDendrogram(RootedDendrogram):
    aspect_distorts_lengths = False

    def ycoords(self, scale, y1):
        cys = [c.y1 for c in self.children]
        if cys:
            y2 = (cys[0]+cys[-1]) / 2.0
        else:
            y2 = y1 - 0.5 * scale.y
        return (y2, y2)

    def xcoords(self, scale, x1):
        if self.is_tip():
            return (x1, (scale.height-(self.height-self.length))*scale.x)
        else:
            # give some margins for internal nodes
            dx = scale.x * self.length * 0.95
            x2 = x1 + dx
            return (x1, x2)

    @classmethod
    def from_tree(cls, tree):
        """ Creates an SquareDendrogram object from a skbio tree.

        Parameters
        ----------
        tree : skbio.TreeNode
            Input skbio tree

        Returns
        -------
        SquareDendrogram
        """
        for n in tree.postorder(include_self=True):
            n.__class__ = SquareDendrogram
        tree.update_geometry(use_lengths=False)
        return tree


class StraightDendrogram(RootedDendrogram):
    def ycoords(self, scale, y1):
        # has a side effect of adjusting the child y1's to meet nodes' y2's
        cys = [c.y1 for c in self.children]
        if cys:
            y2 = (cys[0]+cys[-1]) / 2.0
            distances = [child.length for child in self.children]
            closest_child = self.children[distances.index(min(distances))]
            dy = closest_child.y1 - y2
            max_dy = 0.8*max(5, closest_child.length*scale.x)
            if abs(dy) > max_dy:
                # 'moved', node.Name, y2, 'to within', max_dy,
                # 'of', closest_child.Name, closest_child.y1
                y2 = closest_child.y1 - _sign(dy) * max_dy
        else:
            y2 = y1 - scale.y / 2.0
        y1 = y2
        for child in self.children:
            child.y1 = y2
        return (y1, y2)

    def xcoords(self, scale, x1):
        dx = self.length * scale.x
        dy = self.y2 - self.y1
        dx = numpy.sqrt(max(dx**2 - dy**2, 1))
        return (x1, x1 + dx)

    @classmethod
    def from_tree(cls, tree):
        """ Creates an StraightDendrogram object from a skbio tree.

        Parameters
        ----------
        tree : skbio.TreeNode
            Input skbio tree

        Returns
        -------
        StraightDendrogram
        """
        for n in tree.postorder(include_self=True):
            n.__class__ = StraightDendrogram
        tree.update_geometry(use_lengths=False)
        return tree


class ShelvedDendrogram(RootedDendrogram):
    """A dendrogram in which internal nodes also get a row to themselves
    and the tips are aligned."""

    def __init__(self, use_lengths=False, **kwargs):
        """ Constructs a ShelvedDendrogram object for visualization.

        Parameters
        ----------
        use_lengths: bool
            Specifies if the branch lengths should be included in the
            resulting visualization (default True).
        """
        super().__init__(**kwargs)
        self.use_lengths_default = use_lengths

    def width_required(self):
        # Total number of nodes in the tree.
        return len([n for n in self.levelorder(include_self=True)])

    def xcoords(self, scale, x1):
        """
        Parameters
        ----------
        scale : Dimensions
            Scaled dimensions of the tree
        x1 : int
            X-coordinate of parent

        Returns
        -------
        tuple, int
           x coordinates of parent and child nodes
        """
        return (x1, (scale.height - (self.height - self.length)) * scale.x)

    def ycoords(self, scale, y1):
        """
        Parameters
        ----------
        scale : Dimensions
            Scaled dimensions of the tree
        y1 : int
            Y-coordinate of parent
        Returns
        -------
        tuple, int
           y coordinates of parent and child nodes
        """
        cys = [c.y1 for c in self.children]
        if cys:
            y2 = cys[-1] - 1.0 * scale.y
        else:
            y2 = y1 - 0.5 * scale.y
        return (y2, y2)

    @classmethod
    def from_tree(cls, tree):
        """ Creates an ShelvedDendrogram object from a skbio tree.

        Parameters
        ----------
        tree : skbio.TreeNode
            Input skbio tree

        Returns
        -------
        ShelvedDendrogram
        """
        for n in tree.postorder(include_self=True):
            n.__class__ = ShelvedDendrogram
        tree.update_geometry(use_lengths=False)
        return tree

