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
from q2_composition.plugin_setup import Composition
from q2_types.tree import Phylogeny, Rooted, Unrooted
from gneiss.plugin_setup import plugin
from gneiss.balances import balance_basis
from q2_types.feature_table import FeatureTable
from q2_composition import Balance


def ilr_transform(table: pd.DataFrame, tree: skbio.TreeNode) -> pd.DataFrame:
    basis, _ = balance_basis(tree)
    balances = ilr(table.values, basis)
    in_nodes = [n.name for n in tree.levelorder() if not n.is_tip()]
    return pd.DataFrame(balances,
                        columns=in_nodes,
                        index=table.index)


plugin.methods.register_function(
    function=ilr_transform,
    inputs={'table': FeatureTable[Composition],
            'tree': Phylogeny[Rooted | Unrooted]},
    outputs=[('balances', FeatureTable[Balance])],
    parameters={},
    name='Isometric Log-ratio Transform',
    input_descriptions={
        'table': ('The feature table containing the samples in which '
                  'the ilr transform will be performed.'),
        'tree': ('A hierarchy of feature identifiers that defines the '
                 'partitions of features.  Each tip in the hierarchy'
                 'corresponds to the feature identifiers in the table. '
                 'This tree can contain tip ids that are not present in '
                 'the table, but all feature ids in the table must be '
                 'present in this tree.  This assumes that all of the '
                 'internal nodes in the tree have labels.')
    },
    parameter_descriptions={},
    output_descriptions={'balances': ('The resulting balances from the '
                                      'ilr transform.')},
    description="Calculate balances given a hierarchy."
)
