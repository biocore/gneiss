# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the GPLv3 License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------
from qiime2.plugin import SemanticType
from q2_types.tree import NewickDirectoryFormat
from gneiss.plugin_setup import plugin
from gneiss.q2 import RegressionFormat_g, RegressionDirectoryFormat_g


# Regression types
Regression_g = SemanticType('Regression_g', field_names=['type'])

Linear_g = SemanticType('Linear_g',
                        variant_of=Regression_g.field['type'])
LinearMixedEffects_g = SemanticType('LinearMixedEffects_g',
                                    variant_of=Regression_g.field['type'])

# Tree types
Hierarchy_g = SemanticType('Hierarchy_g', field_names=['type'])


# Define all of the types
plugin.register_semantic_types(
    # Regression types
    Regression_g, Linear_g, LinearMixedEffects_g,
    # Tree type
    Hierarchy_g
)

# Register all of the formats
# Regression types
plugin.register_formats(RegressionFormat_g, RegressionDirectoryFormat_g)
plugin.register_semantic_type_to_format(
    Regression_g[Linear_g | LinearMixedEffects_g],
    artifact_format=RegressionDirectoryFormat
)

# Tree types
plugin.register_semantic_type_to_format(Hierarchy_g,
                                        artifact_format=NewickDirectoryFormat)
