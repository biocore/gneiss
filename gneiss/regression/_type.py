# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------
from qiime2.plugin import SemanticType
from gneiss.plugin_setup import plugin
from ._format import (LinearRegressionDirectoryFormat_g,
                      LinearMixedEffectsDirectoryFormat_g)


LinearRegression_g = SemanticType('LinearRegression_g')
LinearMixedEffects_g = SemanticType('LinearMixedEffects_g')


plugin.register_semantic_types(LinearRegression_g,
                               LinearMixedEffects_g)

plugin.register_semantic_type_to_format(
    LinearRegression_g,
    artifact_format=LinearRegressionDirectoryFormat_g
)

plugin.register_semantic_type_to_format(
    LinearMixedEffects_g,
    artifact_format=LinearMixedEffectsDirectoryFormat_g
)
