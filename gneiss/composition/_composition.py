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

  """ Performs isometric logratio (ilr) transformation on feature-table.
      Zeros must first be removed from the table (e.g. add-pseudocount).
      Creates a new table with balances (groups of features) that distinguish samples. 
  """

def ilr_transform(table: pd.DataFrame, tree: skbio.TreeNode) -> pd.DataFrame:
    _table, _tree = match_tips(table, tree)
    basis, _ = balance_basis(_tree)
    balances = ilr(_table.values, basis)
    in_nodes = [n.name for n in _tree.levelorder() if not n.is_tip()]
    return pd.DataFrame(balances,
                        columns=in_nodes,
                        index=table.index)

