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
import pandas as pd
from .balances import balance_basis
from skbio.stats.composition import ilr
from scipy.cluster.hierarchy import ward
from skbio import TreeNode, DistanceMatrix


def match(table, metadata):
    """ Matches samples between a contingency table and a metadata table.

    Sorts samples in metadata and contingency table in the same order.
    If there are sames contained in the contigency table, but not in metadata
    or vice versa, the intersection of samples in the contingency table and the
    metadata table will returned.

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

    """
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


def _intersect_of_table_metadata_tree(table, metadata, tree):
    """ The intersection of tips, samples and features.

    This calculates the common features between the table and the tree,
    in addition to the common samples between the table and the metadata
    table.  The common subset of features and samples between these
    three objects will be returned.


    Parameters
    ----------
    table : pd.DataFrame
        Contingency table where samples correspond to rows and
        features correspond to columns.
    metadata: pd.DataFrame
        Metadata table that contains information about the samples contained
        in the `table` object.  Samples correspond to rows and covariates
        correspond to columns.
    tree : skbio.TreeNode
        Tree object where the leaves correspond to the columns contained in
        the sample table.

    Returns
    -------
    pd.DataFrame
        Subset of `table` with common row names as `metadata`
        and common columns as `tree.tips()`
    pd.DataFrame
        Subset of `metadata` with common row names as `table`
    skbio.TreeNode
        Subtree of `tree` with common tips as `table`
    """
    if np.any(table <= 0):
        raise ValueError('Cannot handle zeros or negative values in `table`. '
                         'Use pseudocounts or ``multiplicative_replacement``.'
                         )
    # check to see if there are overlapping nodes in tree and table
    overlap = {n.name for n in tree.tips()} & set(table.columns)
    if len(overlap) == 0:
        raise ValueError('There are no internal nodes in `tree` after'
                         'intersection with `table`.')

    _table, _metadata = match(table, metadata)
    _table, _tree = match_tips(_table, tree)
    non_tips_no_name = [(n.name is None) for n in _tree.levelorder()
                        if not n.is_tip()]

    if any(non_tips_no_name):
        _tree = rename_internal_nodes(_tree)
    return _table, _metadata, _tree


def _to_balances(table, tree):
    """ Converts a table of abundances to balances given a tree.

    Parameters
    ----------
    table : pd.DataFrame
        Contingency table where samples correspond to rows and
        features correspond to columns.
    tree : skbio.TreeNode
        Tree object where the leaves correspond to the columns contained in
        the table.

    Returns
    -------
    pd.DataFrame
        Contingency table where samples correspond to rows and
        balances correspond to columns.
    np.array
        Orthonormal basis in the Aitchison simplex generated from `tree`.
    """
    non_tips = [n.name for n in tree.levelorder() if not n.is_tip()]
    basis, _ = balance_basis(tree)

    mat = ilr(table.values, basis=basis)
    ilr_table = pd.DataFrame(mat,
                             columns=non_tips,
                             index=table.index)
    return ilr_table, basis


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
        except:
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
        Table of
    """
    p = n - b + 1  # samples
    y = [1./b] * b + [0] * (n-b)

    table = _shift(y, p-1)
    table = np.column_stack(table)
    return table
