# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the GPLv3 License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------
import pandas as pd
import skbio
from ._ols import OLSModel, ols
from ._mixedlm import LMEModel, mixedlm
from gneiss.util import HAVE_Q2

from q2_composition.plugin_setup import Composition
from q2_types.feature_table import FeatureTable
from q2_types.tree import Phylogeny, Rooted, Unrooted
from qiime2.plugin import Str, Metadata, SemanticType
from gneiss.plugin_setup import plugin
from ._type import Regression_g, Linear_g, LinearMixedEffects_g


def ols_regression(table: pd.DataFrame, tree: skbio.TreeNode,
                   metadata: Metadata, formula: str) -> OLSModel:
    res = ols(table=table, tree=tree, metadata=metadata._dataframe,
              formula=formula)
    res.fit()
    return res


plugin.methods.register_function(
    function=ols_regression,
    inputs={'table': FeatureTable[Composition],
            'tree': Phylogeny[Rooted | Unrooted]},
    parameters={'formula': Str, 'metadata': Metadata},
    outputs=[('linear_model', Regression_g[Linear_g])],
    name='Simplicial Ordinary Least Squares Regression',
    description="Perform linear regression on balances."
)


def lme_regression(table: pd.DataFrame, tree: skbio.TreeNode,
                   metadata: Metadata, formula: str,
                   groups: str) -> LMEModel:
    res = mixedlm(table=table, tree=tree, metadata=metadata._dataframe,
                  formula=formula, groups=groups)
    res.fit()
    return res


plugin.methods.register_function(
    function=lme_regression,
    inputs={'table': FeatureTable[Composition],
            'tree': Phylogeny[Rooted | Unrooted]},
    parameters={'metadata': Metadata, 'formula': Str, 'groups': Str},
    outputs=[('linear_mixed_effects_model',
              Regression_g[LinearMixedEffects_g])],
    name='Simplicial Linear mixed effects regression',
    description="Build and run linear mixed effects model on balances."
)


