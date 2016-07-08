from __future__ import division
import numpy as np
import pandas as pd
from skbio.stats.composition import clr_inv
from collections import OrderedDict
from ete3 import Tree, TreeStyle
from gneiss.layouts import default_layout


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
    nds = list(nds)
    return basis, nds


def balance_basis(tree_node):
    """
    Determines the basis based on binary tree.

    This is commonly referred to as sequential binary partition.
    Given a binary tree relating a list of features, this module can
    be used to calculate an orthonormal basis, which is used to
    calculate the ilr transform.

    Parameters
    ----------
    treenode : skbio.TreeNode
        Binary tree.  Must be a strictly bifurcating tree.

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


def _attach_balances(balances, tree):
    """ Appends the balances to each of the internal nodes
    in the ete tree.

    Parameters
    ----------
    balances : array_like, pd.Series
        Vector of balances to plot on internal nodes of the tree.
        If the balances is not in a `pd.Series`, it is assumed
        to be stored in level order.
    tree : skbio.TreeNode
        Bifurcating tree to plot balances on.

    Return
    ------
    ete.Tree
        The ETE representation of the tree with balances encoded
        as node weights.
    """
    nodes = [n for n in tree.traverse(include_self=True)]
    n_tips = sum([n.is_tip() for n in nodes])
    n_nontips = len(nodes) - n_tips
    if len(balances) != n_nontips:
        raise IndexError('The number of balances (%d) is not '
                         'equal to the number of internal nodes '
                         'in the tree (%d)' % (len(balances), n_nontips))
    ete_tree = Tree.from_skbio(tree)
    # Some random features in all nodes
    i = 0
    for n in ete_tree.traverse():
        if not n.is_leaf():
            if not isinstance(balances, pd.Series):
                n.add_features(weight=balances[i])
            else:
                n.add_features(weight=balances.loc[n.name])
            i += 1
    return ete_tree


def balanceplot(balances, tree,
                layout=None,
                mode='c'):
    """ Plots balances on tree.

    Parameters
    ----------
    balances : np.array
        A vector of internal nodes and their associated real-valued balances.
        The order of the balances will be assumed to be in level order.
    tree : skbio.TreeNode
        A strictly bifurcating tree defining a hierarchical relationship
        between all of the features within `table`.
    layout : function, optional
        A layout for formatting the tree visualization. Must take a
        `ete.tree` as a parameter.
    mode : str
        Type of display to show the tree. ('c': circular, 'r': rectangular).

    Note
    ----
    The `tree` is assumed to strictly bifurcating and whose tips match
    `balances.  It is not recommended to attempt to plot trees with a
    ton of leaves (i.e. more than 4000 leaves).


    Examples
    --------
    >>> from gneiss.balances import balanceplot
    >>> from skbio import TreeNode
    >>> tree = u"((b,c)a, d)root;"
    >>> t = TreeNode.read([tree])
    >>> balances = [10, -10]
    >>> tr, ts = balanceplot(balances, t)
    >>> print(tr.get_ascii())
    <BLANKLINE>
           /-b
        /a|
    -root  \-c
       |
        \-d


    See Also
    --------
    skbio.TreeNode.levelorder
    """
    ete_tree = _attach_balances(balances, tree)

    # Create an empty TreeStyle
    ts = TreeStyle()

    # Set our custom layout function
    if layout is None:
        ts.layout_fn = default_layout
    else:
        ts.layout_fn = layout
    # Draw a tree
    ts.mode = mode

    # We will add node names manually
    ts.show_leaf_name = False
    # Show branch data
    ts.show_branch_length = True
    ts.show_branch_support = True

    return ete_tree, ts
