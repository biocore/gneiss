# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------
import pandas as pd
import skbio
from skbio.stats.composition import ilr
from gneiss.balances import balance_basis
from gneiss.util import match_tips


def ilr_transform(table: pd.DataFrame, tree: skbio.TreeNode) -> pd.DataFrame:
    """Performs isometric logratio (ilr) transformation on feature-table.

    This creates a new table with balances (groups of features) that
    distinguish samples. Zeros must first be removed from the table
    (e.g. add-pseudocount). For source documentation check out:
    https://numpydoc.readthedocs.io/en/latest/

    Parameters
    -----------
    table : pd.DataFrame
        Dataframe of the feature table where rows correspond to samples
        and columns are features. The values within the table must be
        positive and nonzero.
    tree : skbio.TreeNode
        A tree relating all of the features to balances or
        log-contrasts (hierarchy). This tree must be bifurcating
        (i.e. has exactly 2 nodes). The internal nodes of the tree
         will be renamed.

    Returns
    --------
    balances : pd.DataFrame
         Balances calculated from the feature table. Balance represents
         the log ratio of subchildren values below the specified internal node.
    """
    _table, _tree = match_tips(table, tree)
    basis, _ = balance_basis(_tree)
    balances = ilr(_table.values, basis)
    in_nodes = [n.name for n in _tree.levelorder() if not n.is_tip()]
    return pd.DataFrame(balances,
                        columns=in_nodes,
                        index=table.index)
