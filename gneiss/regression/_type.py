# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the GPLv3 License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------
from qiime2.plugin import SemanticType
from gneiss.plugin_setup import plugin
from ._format import RegressionDirectoryFormat_g

Regression_g = SemanticType('Regression_g', field_names=['type'])

Linear_g = SemanticType('Linear_g',
                        variant_of=Regression_g.field['type'])

LinearMixedEffects_g = SemanticType('LinearMixedEffects_g',
                                    variant_of=Regression_g.field['type'])

plugin.register_semantic_types(Regression_g, Linear_g,
                               LinearMixedEffects_g)

plugin.register_semantic_type_to_format(
    Regression_g[Linear_g | LinearMixedEffects_g],
    artifact_format=RegressionDirectoryFormat_g
)
