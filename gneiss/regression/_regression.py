# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the GPLv3 License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------
import numpy as np
import pandas as pd
from skbio.stats.composition import ilr
from gneiss.util import match, match_tips, rename_internal_nodes
from gneiss.balances import balance_basis


def _intersect_of_table_metadata_tree(table, metadata, tree):
    """ Matches tips, features and samples between the table, metadata
    and tree.  This module returns the features and samples that are
    contained in all 3 objects.

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
        the table.

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
    overlap = {n.name for n in tree.traverse()} & set(table.columns)
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

# q2
from ._mixedlm import LinearMixedEffects_g
from ._ols import Linear_g
from ._model import RegressionDirectory_g, RegressionDirectoryFormat_g
from qiime2.plugin import SemanticType

plugin.register_formats(RegressionFormat_g, RegressionDirectoryFormat_g)
plugin.register_semantic_type_to_format(
    Regression_g[Linear_g | LinearMixedEffects_g],
    artifact_format=RegressionDirectoryFormat_g
)
