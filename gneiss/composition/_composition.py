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
from gneiss.plugin_setup import plugin
from gneiss.balances import balance_basis
from gneiss.util import match_tips


def ilr_transform(table: pd.DataFrame, tree: skbio.TreeNode) -> pd.DataFrame:
    _table, _tree = match_tips(table, tree)
    basis, _ = balance_basis(_tree)
    balances = ilr(_table.values, basis)
    in_nodes = [n.name for n in _tree.levelorder() if not n.is_tip()]
    return pd.DataFrame(balances,
                        columns=in_nodes,
                        index=table.index)
