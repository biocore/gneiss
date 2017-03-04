# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------
import pandas as pd
import skbio

from q2_composition.plugin_setup import Composition
from q2_types.feature_table import FeatureTable, Frequency, RelativeFrequency
from q2_types.tree import Phylogeny, Rooted
from qiime2.plugin import MetadataCategory
from gneiss.plugin_setup import plugin
from gneiss.cluster._pba import proportional_linkage, gradient_linkage


def proportional_clustering(table: pd.DataFrame) -> skbio.TreeNode:
    return proportional_linkage(table)


plugin.methods.register_function(
    function=proportional_clustering,
    inputs={'table': FeatureTable[Composition]},
    outputs=[('clustering', Phylogeny[Rooted])],
    name='Hierarchical clustering using proportionality.',
    input_descriptions={
        'table': ('The feature table containing the samples in which '
                  'the columns will be clustered.')},
    parameters={},
    output_descriptions={
        'clustering': ('A hierarchy of feature identifiers where each tip'
                       'corresponds to the feature identifiers in the table. '
                       'This tree can contain tip ids that are not present in '
                       'the table, but all feature ids in the table must be '
                       'present in this tree.')},
    description=('Build a bifurcating tree that represents a hierarchical '
                 'clustering of features.  The hiearchical clustering '
                 'uses Ward hierarchical clustering based on the degree of '
                 'proportionality between features.')
)


def gradient_clustering(table: pd.DataFrame,
                        gradient: MetadataCategory) -> skbio.TreeNode:
    return gradient_linkage(table, gradient.to_series())


plugin.methods.register_function(
    function=gradient_clustering,
    inputs={
        'table': FeatureTable[Frequency | RelativeFrequency | Composition]},
    outputs=[('clustering', Phylogeny[Rooted])],
    name='Hierarchical clustering using gradient information.',
    input_descriptions={
        'table': ('The feature table containing the samples in which '
                  'the columns will be clustered.'),
    },
    parameters={'gradient': MetadataCategory},
    parameter_descriptions={
        'gradient': ('Contains gradient values to sort the '
                     'features and samples.')
    },
    output_descriptions={
        'clustering': ('A hierarchy of feature identifiers where each tip'
                       'corresponds to the feature identifiers in the table. '
                       'This tree can contain tip ids that are not present in '
                       'the table, but all feature ids in the table must be '
                       'present in this tree.')},
    description=('Build a bifurcating tree that represents a hierarchical '
                 'clustering of features.  The hiearchical clustering '
                 'uses Ward hierarchical clustering based on the mean '
                 'difference of gradients that each feature is observed in. '
                 'This method is primarily used to sort the table to reveal '
                 'the underlying block-like structures.')
)
