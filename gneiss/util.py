"""
Utility functions (:mod:`gneiss.util`)
======================================

.. currentmodule:: gneiss.util

This module contains helper functions for aligning metadata tables,
contingency tables and trees.

Functions
---------

.. autosummary::
   :toctree: generated/

   match
   match_tips
   rename_internal_nodes
   block_diagonal
   band_diagonal
"""
# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------
import warnings
import numpy as np
from skbio.stats.composition import closure
import pandas as pd
from patsy import dmatrix
import biom

# Specifies which child is numberator and denominator
NUMERATOR = 1
DENOMINATOR = 0


def split_balance(balance, tree):
    """ Splits a balance into its log ratio components.

    Parameters
    ----------
    balance : pd.Series
        A vector corresponding to a single balance.  These values
        that will be split into its numberator and denominator
        components.

    Returns
    -------
    pd.DataFrame
        Dataframe where the first column contains the numerator and the
        second column contains the denominator of the balance.

    Note
    ----
    The balance must have a name associated with it.
    """
    node = tree.find(balance.name)

    if node.is_tip():
        raise ValueError("%s is not a balance." % balance.name)

    left = node.children[0]
    right = node.children[1]
    if left.is_tip():
        L = 1
    else:
        L = len([n for n in left.tips()])
    if right.is_tip():
        R = 1
    else:
        R = len([n for n in right.tips()])
    b = np.expand_dims(balance.values, axis=1)
    # need to scale down by the number of children in subtrees
    b = np.exp(b / (np.sqrt((L*R) / (L + R))))
    o = np.ones((len(b), 1))
    k = np.hstack((b, o))
    p = closure(k)
    return pd.DataFrame(p, columns=[left.name, right.name],
                        index=balance.index)


def match(table, metadata):
    """ Matches samples between a contingency table and a metadata table.

    Sorts samples in metadata and contingency table in the same order.
    If there are sames contained in the contigency table, but not in metadata
    or vice versa, the intersection of samples in the contingency table and the
    metadata table will returned.

    Parameters
    ----------
    table : pd.DataFrame or biom.Table
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
    """
    if isinstance(table, pd.DataFrame):
        return _dense_match(table, metadata)
    elif isinstance(table, biom.Table):
        return _sparse_match(table, metadata)


def _dense_match(table, metadata):
    """ Match on dense pandas tables"""
    subtableids = set(table.index)
    submetadataids = set(metadata.index)
    if len(subtableids) != len(table.index):
        raise ValueError("`table` has duplicate sample ids.")
    if len(submetadataids) != len(metadata.index):
        raise ValueError("`metadata` has duplicate sample ids.")

    idx = subtableids & submetadataids
    if len(idx) == 0:
        raise ValueError(("No more samples left.  Check to make sure that "
                          "the sample names between `metadata` and `table` "
                          "are consistent"))

    subtable = table.loc[idx]
    submetadata = metadata.loc[idx]
    return subtable, submetadata


def _sparse_match(table, metadata):
    """ Match on sparse biom tables. """
    subtableids = set(table.ids(axis='sample'))
    submetadataids = set(metadata.index)
    if len(submetadataids) != len(metadata.index):
        raise ValueError("`metadata` has duplicate sample ids.")
    idx = subtableids & submetadataids
    if len(idx) == 0:
        raise ValueError(("No more samples left.  Check to make sure that "
                          "the sample names between `metadata` and `table` "
                          "are consistent"))

    out_metadata = metadata.loc[idx]

    def metadata_filter(val, id_, md):
        return id_ in out_metadata.index

    out_table = table.filter(metadata_filter, axis='sample', inplace=False)

    def sort_f(xs):
        return [xs[out_metadata.index.get_loc(x)] for x in xs]

    out_table = out_table.sort(sort_f=sort_f, axis='sample')
    return out_table, out_metadata


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
    table : pd.DataFrame or biom.Table
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
    if isinstance(table, pd.DataFrame):
        return _dense_match_tips(table, tree)
    elif isinstance(table, biom.Table):
        return _sparse_match_tips(table, tree)


def _sparse_match_tips(table, tree):
    """ Match on sparse biom tables. """
    tips = [x.name for x in tree.tips()]
    common_tips = set(tips) & set(table.ids(axis='observation'))

    _tree = tree.shear(names=list(common_tips))

    def filter_uncommon(val, id_, md):
        return id_ in common_tips
    _table = table.filter(filter_uncommon, axis='observation', inplace=False)

    _tree.bifurcate()
    _tree.prune()

    def sort_f(x):
        return [n.name for n in _tree.tips()]

    _table = _table.sort(sort_f=sort_f, axis='observation')
    return _table, _tree


def _dense_match_tips(table, tree):
    """ Match on dense pandas dataframes. """
    tips = [x.name for x in tree.tips()]
    common_tips = list(set(tips) & set(table.columns))
    _table = table.loc[:, common_tips]
    _tree = tree.shear(names=common_tips)

    _tree.bifurcate()
    _tree.prune()
    sorted_features = [n.name for n in _tree.tips()]
    _table = _table.reindex_axis(sorted_features, axis=1)
    return _table, _tree


def design_formula(train_metadata, test_metadata, formula):
    """ Generate and align two design matrices.

    Parameters
    ----------
    train_metadata : pd.DataFrame
        Training metadata
    test_metadata : pd.DataFrame
        Testing metadata
    formula : str
        Statistical formula specifying design matrix

    Return
    ------
    train_design : pd.DataFrame
        Train design matrix
    test_design : pd.DataFrame
        Test design matrix
    """
    train_design = dmatrix(formula, train_metadata,
                           return_type='dataframe')
    test_design = dmatrix(formula, test_metadata,
                          return_type='dataframe')

    # pad extra columns with zeros, so that we can still make predictions
    extra_columns = list(set(train_design.columns) -
                         set(test_design.columns))
    df = pd.DataFrame({C: np.zeros(test_design.shape[0])
                       for C in extra_columns},
                      index=test_design.index)
    test_design = pd.concat((test_design, df), axis=1)
    test_design = test_design.reindex(columns=train_design.columns)
    return train_design, test_design


def check_internal_nodes(tree):
    for n in tree.levelorder():
        if n.name is None:
            raise ValueError('TreeNode has no name.')


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

    Raises
    ------
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
                              "with (%s)" % (n.name, label), UserWarning)

            n.name = label
            i += 1
    return _tree


def _type_cast_to_float(df):
    """ Attempt to cast all of the values in dataframe to float.

    This will try to type cast all of the series within the
    dataframe into floats.  If a column cannot be type casted,
    it will be kept as is.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    # TODO: Will need to improve this, as this is a very hacky solution.
    for c in df.columns:
        s = df[c]
        try:
            df[c] = s.astype(np.float64)
        except Exception:
            continue
    return df


def block_diagonal(ncols, nrows, nblocks):
    """ Generate block diagonal with uniformly distributed values within blocks.

    Parameters
    ----------
    ncol : int
        Number of columns
    nrows : int
        Number of rows
    nblocks : int
        Number of blocks

    Returns
    -------
    np.array
        Table with a block diagonal where the rows represent samples
        and the columns represent features.  The values within the blocks
        are uniformly distributed between 0 and 1.

    Note
    ----
    The number of blocks specified by `nblocks` needs to be greater than 1.
    """
    if nblocks <= 1:
        raise ValueError('`nblocks` needs to be greater than 1.')
    mat = np.zeros((nrows, ncols))
    block_cols = ncols // nblocks
    block_rows = nrows // nblocks
    for b in range(nblocks-1):
        B = np.random.uniform(size=(block_rows, block_cols))
        lower_row = block_rows * b
        upper_row = min(block_rows*(b+1), nrows)
        lower_col = block_cols * b
        upper_col = min(block_cols*(b+1), ncols)

        mat[lower_row:upper_row, lower_col:upper_col] = B

    # Make last block fill in the remainder
    B = np.random.uniform(size=(nrows-upper_row, ncols-upper_col))
    mat[upper_row:, upper_col:] = B
    return mat


def _shift(l, n):
    """ Creates the band table by iteratively shifting a single vector.

    Parameters
    ----------
    l : array
       Vector to be shifted
    n : int
       Max number of shifts
    """
    sl = l

    table = [l]

    if n == 0:
        return table
    else:
        for k in range(n):
            sl = np.roll(sl, 1)
            table.append(sl)
        return table


def band_diagonal(n, b):
    """ Creates band table with dense diagonal, sparse corners.

    Parameters
    ----------
    n : int
        Number of features
    b : int
        Length of band

    Returns
    -------
    np.array
        Table with a dense band diagonal where the rows represent samples
        and the columns represent features.  The values within the
        diagonal are marked with a constant `1/b`.
    """
    p = n - b + 1  # samples
    y = [1./b] * b + [0] * (n-b)

    table = _shift(y, p-1)
    table = np.column_stack(table)
    return table
