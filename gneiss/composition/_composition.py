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
from gneiss.util import match_tips, rename_internal_nodes


def ilr_hierarchical(table: pd.DataFrame, tree: skbio.TreeNode) -> (
                     pd.DataFrame):
    _table, _tree = match_tips(table, tree)
    basis, _ = balance_basis(_tree)
    balances = ilr(_table.values, basis)
    in_nodes = [n.name for n in _tree.levelorder() if not n.is_tip()]
    return pd.DataFrame(balances,
                        columns=in_nodes,
                        index=table.index)


def ilr_phylogenetic(table: pd.DataFrame, tree: skbio.TreeNode) -> (
                     skbio.TreeNode, pd.DataFrame):
    t = tree.copy()
    t.bifurcate()
    t = rename_internal_nodes(t)
    _table, _tree = match_tips(table, t)
    basis, _ = balance_basis(_tree)
    balances = ilr(_table.values, basis)
    in_nodes = [n.name for n in _tree.levelorder() if not n.is_tip()]
    return t, pd.DataFrame(balances,
                           columns=in_nodes,
                           index=table.index)
