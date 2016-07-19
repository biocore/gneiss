import pandas as pd
import numpy as np


def match(x, y, intersect=False):
    """ Sorts samples in metadata and contingency table in the same order.

    Parameters
    ----------
    x : pd.DataFrame
        Contingency table where samples correspond to rows and
        features correspond to columns.
    y: pd.DataFrame
        Metadata table where samples correspond to rows and
        explanatory metadata variables correspond to columns.
    intersect : bool, optional
        Specifies if only the intersection of samples in the
        contingency table and the metadata table will returned.

    Returns
    -------
    _x : pd.DataFrame
        Filtered dataframe
    _y : pd.DataFrame
        Filtered dataframe
    """
    _x = x.sort_index()
    _y = y.sort_index()
    if intersect:
        idx = set(_x.index) & set(_y.index)
        idx = sorted(idx)
        return _x.loc[idx], _y.loc[idx]
    else:
        if len(_x.index) != len(_y.index):
            raise ValueError("`x` and `y` have incompatible sizes, "
                             "`x` has %d rows, `y` has %d rows.  "
                             "Consider setting `intersect=True`." %
                             (len(_x.index), len(_y.index)))
        return _x, _y


def match_tips(table, tree, intersect=False):
    """ Returns the OTU table and tree with matched tips.

    Sorts the columns of the OTU table to match the tips in
    the tree.  If the tree is multi-furcating, then the
    tree is reduced to a bifurcating tree by randomly inserting
    internal nodes.


    Parameters
    ----------
    table : pd.DataFrame
        Contingency table where samples correspond to rows and
        features correspond to columns.
    tree : skbio.TreeNode
        Tree object where the leafs correspond to the features.
    intersect : bool, optional
        Specifies if only the intersection of samples in the
        contingency table and the tree will returned.

    Returns
    -------
    pd.DataFrame :
        Subset of the original contingency table with the common features.
    skbio.TreeNode :
        Sub-tree with the common features.
    """
    tips = [x.name for x in tree.tips()]
    common_tips = list(set(tips) & set(table.columns))

    if intersect:
        _table = table.loc[:, common_tips]
        _tree = tree.shear(names=common_tips)
    else:
        if len(tips) != len(table.columns):
            raise ValueError("`table` and `tree` have incompatible sizes, "
                             "`table` has %d columns, `tree` has %d tips.  "
                             "Consider setting `intersect=True`." %
                             (len(table.columns), len(tips)))

        _table = table
        _tree = tree

    _tree.bifurcate()
    _tree.prune()
    sorted_features = [n.name for n in _tree.tips()]
    _table = _table.reindex_axis(sorted_features, axis=1)
    return _table, _tree


def rename_tips(tree, names=None):
    """ Names the tree tips according to level ordering.

    The tree will be traversed from top-down, left to right.
    If there `names` is not specified, the node with the smallest label (y0)
    will be located at the root of the tree, and the node with the largest
    label will be located at bottom right corner of the tree.

    Parameters
    ----------
    tree : skbio.TreeNode
        Tree object where the leafs correspond to the features.
    names : list, optional
        List of labels to rename the tip names.  It is assumed that the
        names are listed in level ordering.

    Returns
    -------
    skbio.TreeNode
       Tree with renamed internal nodes.
    """
    i = 0
    for n in tree.levelorder():
        if not n.is_tip():
            if names is None:
                n.name = 'y%i' % i
            else:
                n.name = names[i]
            i+=1
    return tree
