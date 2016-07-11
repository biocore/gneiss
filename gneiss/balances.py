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
    # TODO: use recarray
    # col 0 -> right counts
    # col 1 -> left counts
    # col 2 -> k
    # col 3 -> t
    r_idx = 0
    l_idx = 1
    k_idx = 2
    t_idx = 3

    counts, n_tips, n_nodes = _count_matrix(tree_node)
    r = counts[:, r_idx]
    s = counts[:, l_idx]
    k = counts[:, k_idx]
    t = counts[:, t_idx]

    a = np.sqrt(s / (r*(r+s)))
    b = -1*np.sqrt(r / (s*(r+s)))

    basis = np.zeros((n_tips-1, n_tips))
    for i in np.arange(n_nodes - n_tips, dtype=int):
        v = basis[i]

        k_i = n_tips - k[i]
        r_i = k_i - r[i]
        s_i = r_i - s[i]

        v[r_i:k_i] = a[i]
        v[s_i:r_i] = b[i]

    return basis, [n for n in tree_node.levelorder() if not n.is_tip()]


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
    node_count = 0
    for n in treenode.postorder(include_self=True):
        node_count += 1
        if n.is_tip():
            n._tip_count = 1
        else:
            try:
                left, right = n.children
            except:
                raise ValueError("Not a strictly bifurcating tree!")
            n._tip_count = left._tip_count + right._tip_count

    # TODO: use recarray
    # col 0 -> right counts
    # col 1 -> left counts
    # col 2 -> k
    # col 3 -> t
    r_idx = 0
    l_idx = 1
    k_idx = 2
    t_idx = 3
    counts = np.zeros((node_count, 4), dtype=int)

    for i, n in enumerate(treenode.levelorder(include_self=True)):
        if n.is_tip():
            continue

        n._lo_idx = i
        node_counts = counts[i]

        node_counts[r_idx] = 1 if n.is_tip() else n.children[1]._tip_count
        node_counts[l_idx] = 1 if n.is_tip() else n.children[0]._tip_count

        if n.is_root():
            k = 0
            t = 0
        else:
            parent_counts = counts[n.parent._lo_idx]
            if n is n.parent.children[0]:
                #t = parent_counts[t_idx] + parent_counts[l_idx]
                #k = parent_counts[k_idx]

                k = parent_counts[k_idx] + parent_counts[r_idx]
                t = parent_counts[t_idx]
            else:
                #k = parent_counts[k_idx] + parent_counts[r_idx]
                #t = parent_counts[t_idx]

                k = parent_counts[k_idx]
                t = parent_counts[t_idx] + parent_counts[l_idx]

        node_counts[k_idx] = k
        node_counts[t_idx] = t

        counts[i] = node_counts

    return counts, treenode._tip_count, node_count


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
