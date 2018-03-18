"""
Balances (:mod:`gneiss.balances`)

=================================

.. currentmodule:: gneiss.balances

This module contains modules for calculating balances and creating ETE
objects to visualize these balances on a tree.

Functions
---------

.. autosummary::
   :toctree: generated/

   balance_basis

"""
# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------


from __future__ import division
import numpy as np
from skbio.stats.composition import clr_inv
from collections import OrderedDict
from gneiss.util import NUMERATOR, DENOMINATOR
from scipy.sparse import coo_matrix


def _balance_basis(tree_node):
    """ Helper method for calculating balance basis
    """
    counts, n_tips = _count_matrix(tree_node)
    counts = OrderedDict([(x, counts[x])
                          for x in counts.keys() if not x.is_tip()])
    nds = counts.keys()
    r = np.array([counts[n]['r'] for n in nds])
    s = np.array([counts[n]['l'] for n in nds])
    k = np.array([counts[n]['k'] for n in nds])
    t = np.array([counts[n]['t'] for n in nds])

    a = np.sqrt(s / (r*(r+s)))
    b = -1*np.sqrt(r / (s*(r+s)))

    basis = np.zeros((n_tips-1, n_tips))
    for i in range(len(nds)):
        basis[i, :] = np.array([0]*k[i] + [a[i]]*r[i] + [b[i]]*s[i] + [0]*t[i])
    # Make sure that the basis is in level order
    basis = basis[:, ::-1]
    nds = [n.name for n in nds]
    return basis, nds


def balance_basis(tree_node):
    """
    Determines the basis based on bifurcating tree.

    This is commonly referred to as sequential binary partition [1]_.
    Given a binary tree relating a list of features, this module can
    be used to calculate an orthonormal basis, which is used to
    calculate the ilr transform.

    Parameters
    ----------
    treenode : skbio.TreeNode
        Input bifurcating tree.  Must be strictly bifurcating
        (i.e. every internal node needs to have exactly 2 children).

    Returns
    -------
    basis : np.array
        Returns a set of orthonormal bases in the Aitchison simplex
        corresponding to the tree. The order of the
        basis is index by the level order of the internal nodes.
    nodes : list, skbio.TreeNode
        List of tree nodes indicating the ordering in the basis.

    Raises
    ------
    ValueError
        The tree doesn't contain two branches.

    Examples
    --------
    >>> from gneiss.balances import balance_basis
    >>> from skbio import TreeNode
    >>> tree = u"((b,c)a, d)root;"
    >>> t = TreeNode.read([tree])
    >>> basis, nodes = balance_basis(t)
    >>> basis
    array([[ 0.18507216,  0.18507216,  0.62985567],
           [ 0.14002925,  0.57597535,  0.28399541]])

    Notes
    -----
    The tree must be strictly bifurcating, meaning that
    every internal node has exactly 2 children.

    See Also
    --------
    skbio.stats.composition.ilr

    References
    ----------
    .. [1] J.J. Egozcue and V. Pawlowsky-Glahn "Exploring Compositional Data
        with the CoDa-Dendrogram" (2011)

    """
    basis, nodes = _balance_basis(tree_node)
    basis = clr_inv(basis)
    return basis, nodes


def _count_matrix(treenode):
    n_tips = 0
    nodes = list(treenode.levelorder(include_self=True))
    # fill in the Ordered dictionary. Note that the
    # elements of this Ordered dictionary are
    # dictionaries.
    counts = OrderedDict()
    columns = ['k', 'r', 'l', 't', 'tips']
    for n in nodes:
        if n not in counts:
            counts[n] = {}
        for c in columns:
            counts[n][c] = 0

    # fill in r and l.  This is done in reverse level order.
    for n in nodes[::-1]:
        if n.is_tip():
            counts[n]['tips'] = 1
            n_tips += 1
        elif len(n.children) == 2:
            lchild = n.children[0]
            rchild = n.children[1]
            counts[n]['r'] = counts[rchild]['tips']
            counts[n]['l'] = counts[lchild]['tips']
            counts[n]['tips'] = counts[n]['r'] + counts[n]['l']
        else:
            raise ValueError("Not a strictly bifurcating tree!")

    # fill in k and t
    for n in nodes:
        if n.parent is None:
            counts[n]['k'] = 0
            counts[n]['t'] = 0
            continue
        elif n.is_tip():
            continue
        # left or right child
        # left = 0, right = 1
        child_idx = 'l' if n.parent.children[0] != n else 'r'
        if child_idx == 'l':
            counts[n]['t'] = counts[n.parent]['t'] + counts[n.parent]['l']
            counts[n]['k'] = counts[n.parent]['k']
        else:
            counts[n]['k'] = counts[n.parent]['k'] + counts[n.parent]['r']
            counts[n]['t'] = counts[n.parent]['t']
    return counts, n_tips


def sparse_balance_basis(tree):
    """ Calculates sparse representation of an ilr basis from a tree.

    This computes an orthonormal basis specified from a bifurcating tree.

    Parameters
    ----------
    tree : skbio.TreeNode
        Input bifurcating tree.  Must be strictly bifurcating
        (i.e. every internal node needs to have exactly 2 children).
        This is used to specify the ilr basis.

    Returns
    -------
    scipy.sparse.coo_matrix
       The ilr basis required to perform the ilr_inv transform.
       This is also known as the sequential binary partition.
       Note that this matrix is represented in clr coordinates.
    nodes : list, str
        List of tree nodes indicating the ordering in the basis.

    Raises
    ------
    ValueError
        The tree doesn't contain two branches.

    """
    # this is inspired by @wasade in
    # https://github.com/biocore/gneiss/pull/8
    t = tree.copy()
    D = len(list(tree.tips()))
    # calculate number of tips under each node
    for n in t.postorder(include_self=True):
        if n.is_tip():
            n._tip_count = 1
        else:
            if len(n.children) == 2:
                left, right = n.children[NUMERATOR], n.children[DENOMINATOR],
            else:
                raise ValueError("Not a strictly bifurcating tree.")
            n._tip_count = left._tip_count + right._tip_count

    # calculate k, r, s, t coordinate for each node
    left, right = t.children[NUMERATOR], t.children[DENOMINATOR],
    t._k, t._r, t._s, t._t = 0, left._tip_count, right._tip_count, 0
    for n in t.preorder(include_self=False):
        if n.is_tip():
            n._k, n._r, n._s, n._t = 0, 0, 0, 0

        elif n == n.parent.children[NUMERATOR]:
            n._k = n.parent._k
            n._r = n.children[NUMERATOR]._tip_count
            n._s = n.children[DENOMINATOR]._tip_count
            n._t = n.parent._s + n.parent._t
        elif n == n.parent.children[DENOMINATOR]:
            n._k = n.parent._r + n.parent._k
            n._r = n.children[NUMERATOR]._tip_count
            n._s = n.children[DENOMINATOR]._tip_count
            n._t = n.parent._t
        else:
            raise ValueError("Tree topology is not correct.")

    # navigate through tree to build the basis in a sparse matrix form
    value = []
    row, col = [], []
    nodes = []
    i = 0

    for n in t.levelorder(include_self=True):

        if n.is_tip():
            continue

        for j in range(n._k, n._k + n._r):
            row.append(i)
            # consider tips in reverse order. May want to rethink
            # this orientation in the future.
            col.append(D-1-j)
            A = np.sqrt(n._s / (n._r * (n._s + n._r)))

            value.append(A)

        for j in range(n._k + n._r, n._k + n._r + n._s):
            row.append(i)
            col.append(D-1-j)
            B = -np.sqrt(n._r / (n._s * (n._s + n._r)))

            value.append(B)
        i += 1
        nodes.append(n.name)

    basis = coo_matrix((value, (row, col)), shape=(D-1, D))

    return basis, nodes
