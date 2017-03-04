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

from q2_composition.plugin_setup import Composition
from q2_types.feature_table import FeatureTable
from q2_types.tree import Phylogeny, Rooted, Unrooted
from qiime2.plugin import Str, Metadata
from gneiss.plugin_setup import plugin
from ._type import LinearRegression_g, LinearMixedEffects_g


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
    outputs=[('linear_model', LinearRegression_g)],
    name='Simplicial Ordinary Least Squares Regression',
    input_descriptions={
        'table': ('The feature table containing the samples in which '
                  'simplicial regression will be performed.'),
        'tree': ('A hierarchy of feature identifiers where each tip'
                 'corresponds to the feature identifiers in the table. '
                 'This tree can contain tip ids that are not present in '
                 'the table, but all feature ids in the table must be '
                 'present in this tree.')
    },
    parameter_descriptions={
        'formula': 'Statistical formula specifying the statistical model.',
        'metadata': ('Metadata information that contains the '
                     'covariates of interest.')
    },
    output_descriptions={'linear_model': ('The resulting '
                                          'fit.')},
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
              LinearMixedEffects_g)],
    name='Simplicial Linear mixed effects regression',
    input_descriptions={
        'table': ('The feature table containing the samples in which '
                  'simplicial regression with mixed effects will be performed'
                  'will be performed.'),
        'tree': ('A hierarchy of feature identifiers where each tip'
                 'corresponds to the feature identifiers in the table. '
                 'This tree can contain tip ids that are not present in '
                 'the table, but all feature ids in the table must be '
                 'present in this tree.')
    },
    parameter_descriptions={
        'formula': 'Statistical formula specifying the statistical model.',
        'metadata': ('Metadata information that contains the '
                     'covariates of interest.')
    },
    output_descriptions={'linear_mixed_effects_model': ('The resulting '
                                                        'fit.')},
    description="Build and run linear mixed effects model on balances."
)
