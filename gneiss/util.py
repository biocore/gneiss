# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the GPLv3 License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------
import warnings


def match(table, metadata):
    """ Sorts samples in metadata and contingency table in the same order.

    The intersection of samples in the contingency table and the metadata table
    will returned.

    Parameters
    ----------
    table : pd.DataFrame
        Contingency table where samples correspond to rows and
        features correspond to columns.
    metadata: pd.DataFrame
        Metadata table where samples correspond to rows and
        explanatory metadata variables correspond to columns.

    Returns
    -------
    pd.DataFrame :
        Filtered contingency table.
    pd.DataFrame :
        Filtered metadata table

    Raises
    ------
    ValueError:
        Raised if duplicate sample ids are present in `table`.
    ValueError:
        Raised if duplicate sample ids are present in `metadata`.
    ValueError:
        Raised if `table` and `metadata` have incompatible sizes.

    Note
    ----
    If `intersect=True` is specified, then the rows for `table` and
    `metadata` will be matched, but they will be in a random order.
    """
    subtableids = set(table.index)
    submetadataids = set(metadata.index)
    if len(subtableids) != len(table.index):
        raise ValueError("`table` has duplicate sample ids.")
    if len(submetadataids) != len(metadata.index):
        raise ValueError("`metadata` has duplicate sample ids.")

    idx = subtableids & submetadataids
    subtable = table.loc[idx]
    submetadata = metadata.loc[idx]
    return subtable, submetadata


def match_tips(table, tree):
    """ Returns the contingency table and tree with matched tips.

    Sorts the columns of the contingency table to match the tips in
    the tree.  The ordering of the tips is in post-traversal order.

    If the tree is multi-furcating, then the tree is reduced to a
    bifurcating tree by randomly inserting internal nodes.

    The intersection of samples in the contingency table and the
    tree will returned.

    Parameters
    ----------
    table : pd.DataFrame
        Contingency table where samples correspond to rows and
        features correspond to columns.
    tree : skbio.TreeNode
        Tree object where the leafs correspond to the features.

    Returns
    -------
    pd.DataFrame :
        Subset of the original contingency table with the common features.
    skbio.TreeNode :
        Sub-tree with the common features.

    Raises
    ------
    ValueError:
        Raised if `table` and `tree` have incompatible sizes.

    See Also
    --------
    skbio.TreeNode.bifurcate
    skbio.TreeNode.tips
    """
    tips = [x.name for x in tree.tips()]
    common_tips = list(set(tips) & set(table.columns))

    _table = table.loc[:, common_tips]
    _tree = tree.shear(names=common_tips)

    _tree.bifurcate()
    _tree.prune()
    sorted_features = [n.name for n in _tree.tips()]
    _table = _table.reindex_axis(sorted_features, axis=1)

    return _table, _tree


def rename_internal_nodes(tree, names=None, inplace=False):
    """ Names the internal according to level ordering.

    The tree will be traversed in level order (i.e. top-down, left to right).
    If `names` is not specified, the node with the smallest label (y0)
    will be located at the root of the tree, and the node with the largest
    label will be located at bottom right corner of the tree.

    Parameters
    ----------
    tree : skbio.TreeNode
        Tree object where the leafs correspond to the features.
    names : list, optional
        List of labels to rename the tip names.  It is assumed that the
        names are listed in level ordering, and the length of the list
        is at least as long as the number of internal nodes.
    inplace : bool, optional
        Specifies if the operation should be done on the original tree or not.

    Returns
    -------
    skbio.TreeNode
       Tree with renamed internal nodes.

    ValueError:
        Raised if `tree` and `name` have incompatible sizes.
    """
    if inplace:
        _tree = tree
    else:
        _tree = tree.copy()

    non_tips = [n for n in _tree.levelorder() if not n.is_tip()]
    if names is not None and len(non_tips) != len(names):
        raise ValueError("`_tree` and `names` have incompatible sizes, "
                         "`_tree` has %d tips, `names` has %d elements." %
                         (len(non_tips), len(names)))

    i = 0
    for n in _tree.levelorder():
        if not n.is_tip():
            if names is None:
                label = 'y%i' % i
            else:
                label = names[i]
            if n.name is not None and label == n.name:
                warnings.warn("Warning. Internal node (%s) has been replaced "
                              "with (%s)" % (n.name, label))

            n.name = label
            i += 1
    return _tree
