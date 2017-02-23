# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the GPLv3 License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------
from gneiss.util import HAVE_Q2
from qiime2.plugin import SemanticType

if HAVE_Q2:
    from ._model import (Regression_g, RegressionFormat_g,
                         RegressionDirectoryFormat_g)
    from gneiss.plugin_setup import plugin

    Linear_g = SemanticType('Linear_g',
                            variant_of=Regression_g.field['type'])

    LinearMixedEffects_g = SemanticType('LinearMixedEffects_g',
                                        variant_of=Regression_g.field['type'])

    plugin.register_semantic_types(Linear_g, LinearMixedEffects_g)
    plugin.register_semantic_type_to_format(
        Regression_g[Linear_g | LinearMixedEffects_g],
        artifact_format=RegressionDirectoryFormat_g
    )
