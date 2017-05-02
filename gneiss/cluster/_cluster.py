# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------
import uuid
import pandas as pd
import numpy as np
import skbio

from q2_composition.plugin_setup import Composition
from q2_types.feature_table import FeatureTable, Frequency, RelativeFrequency
from q2_types.tree import Phylogeny, Rooted
from qiime2.plugin import MetadataCategory, Bool
from gneiss.plugin_setup import plugin
from gneiss.cluster._pba import correlation_linkage, gradient_linkage
from gneiss.sort import gradient_sort, mean_niche_estimator
from gneiss.util import rename_internal_nodes


def correlation_clustering(table: pd.DataFrame) -> skbio.TreeNode:
    """ Builds a tree for features based on correlation.

    Parameters
    ----------
    table : pd.DataFrame
       Contingency table where rows are samples and columns are features.
       In addition, the table must have strictly nonzero values.

    Returns
    -------
    skbio.TreeNode
       Represents the partitioning of features with respect to correlation.
    """
    t = correlation_linkage(table)
    return t


plugin.methods.register_function(
    function=correlation_clustering,
    inputs={'table': FeatureTable[Composition]},
    outputs=[('clustering', Phylogeny[Rooted])],
    name='Hierarchical clustering using feature correlation.',
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
                        gradient: MetadataCategory,
                        weighted=True) -> skbio.TreeNode:
    """ Builds a tree for features based on a gradient.

    Parameters
    ----------
    table : pd.DataFrame
       Contingency table where rows are samples and columns are features.
    gradient : qiime2.MetadataCategory
       Continuous vector of measurements corresponding to samples.
    weighted : bool
       Specifies if abundance or presence/absence information
       should be used to perform the clustering.

    Returns
    -------
    skbio.TreeNode
       Represents the partitioning of features with respect to the gradient.
    """
    c = gradient.to_series()
    c = c.astype(np.float)
    if not weighted:
        table = table > 0

    t = gradient_linkage(table, c, method='average')
    mean_g = mean_niche_estimator(table, c)
    mean_g = pd.Series(mean_g, index=table.columns)
    mean_g = mean_g.sort_values()
    t = gradient_sort(t, mean_g)
    return t


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
    parameters={'gradient': MetadataCategory, 'weighted': Bool},
    parameter_descriptions={
        'gradient': ('Contains gradient values to sort the '
                     'features and samples.'),
        'weighted': ('Specifies if abundance or presence/absence '
                     'information should be used to perform the clustering.'),
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


def assign_ids(tree: skbio.TreeNode) -> skbio.TreeNode:
    def replace_dashes(x):
        return str(x).replace('-', '')
    t = tree.copy()
    t.bifurcate()
    ids = ['%sL_%s' % (i, replace_dashes(uuid.uuid4()))
           for i, n in enumerate(t.levelorder(include_self=True))
           if not n.is_tip()]
    t = rename_internal_nodes(t, names=ids)
    return t


plugin.methods.register_function(
    function=assign_ids,
    inputs={'tree': Phylogeny[Rooted]},
    outputs=[('tree', Phylogeny[Rooted])],
    name='Assigns ids on internal nodes in the tree.',
    input_descriptions={
        'tree': ('The input tree with potential missing ids.')},
    parameters={},
    output_descriptions={
        'tree': ('A tree with uniquely identifying ids.')},
    description=('Assigns UUIDs to uniquely identify internal nodes '
                 'in the tree.  Also corrects for polytomies to create '
                 'strictly bifurcating trees.')
)
