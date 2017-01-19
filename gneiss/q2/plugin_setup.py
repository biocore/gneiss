# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the GPLv3 License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------
import qiime.plugin

import gneiss
import q2_composition
# These imports are only included to support the example methods and
# visualizers. Remove these imports when you are ready to develop your plugin.

from q2_types import Phylogeny, Rooted, FeatureTable

from qiime.plugin import Plugin, Metadata, Str
from gneiss.regression import ols, mixedlm
from q2_composition.plugin_setup import Composition
from gneiss.q2._type import (Regression, Linear, LinearMixedEffects,
                             Hierarchy, Cluster)
from gneiss.q2._format import RegressionDirectoryFormat, RegressionFormat

from q2_types.tree import NewickDirectoryFormat

plugin = qiime.plugin.Plugin(
    name='gneiss',
    version=gneiss.__version__,
    website='https://biocore.github.io/gneiss/',
    package='gneiss',
    # Information on how to obtain user support should be provided as a free
    # text string via user_support_text. If None is provided, users will
    # be referred to the plugin's website for support.
    user_support_text=None,
    # Information on how the plugin should be cited should be provided as a
    # free text string via citation_text. If None is provided, users
    # will be told to use the plugin's website as a citation.
    citation_text=("Morton JT, Sanders J, Quinn RA, McDonald D, "
                   "Gonzalez A, Vázquez-Baeza Y, Navas-Molina JA, "
                   "Song SJ, Metcalf JL, Hyde ER, Lladser M, Dorrestein PC,"
                   " Knight R. 2017. Balance trees reveal microbial niche "
                   "differentiation. mSystems 2:e00162-16. "
                   "https://doi.org/10.1128/mSystems.00162-16.")
)

# Define all of the types
plugin.register_semantic_types(
    # Regression types
    Regression, Linear, LinearMixedEffects,
    # Tree types
    Hierarchy, Cluster
)

# Register all of the formats
# Regression types
plugin.register_formats(RegressionFormat, RegressionDirectoryFormat)
plugin.register_semantic_type_to_format(
    Regression[Linear | LinearMixedEffects],
    artifact_format=RegressionDirectoryFormat
)
# Tree types
plugin.register_semantic_type_to_format(Hierarchy[Cluster],
                                        artifact_format=NewickDirectoryFormat)

# Define all of the main functions
plugin.methods.register_function(
    function=gneiss.regression.ols,
    inputs={'table': FeatureTable[Composition],
            'tree': Hierarchy[Cluster]},
    parameters={'metadata': Metadata},
    outputs=[('linear_model', Regression[Linear])],
    name='Linear Regression',
    description="Perform linear regression on balances."
)

plugin.methods.register_function(
    function=gneiss.regression.mixedlm,
    inputs={'table': FeatureTable[Composition],
            'tree': Hierarchy[Cluster]},
    parameters={'metadata': Metadata},
    outputs=[('linear_mixed_effects_model', Regression[LinearMixedEffects])],
    name='Linear mixed effects models',
    description="Perform linear mixed effects model on balances."
)

