# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the GPLv3 License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------
from gneiss.util import HAVE_Q2
if HAVE_Q2:
    from ._mixedlm import LinearMixedEffects_g
    from ._ols import Linear_g
    from ._model import (Regression_g, RegressionFormat_g,
                         RegressionDirectoryFormat_g)
    from gneiss.plugin_setup import plugin

    plugin.register_formats(RegressionFormat_g, RegressionDirectoryFormat_g)
    plugin.register_semantic_type_to_format(
        Regression_g[Linear_g | LinearMixedEffects_g],
        artifact_format=RegressionDirectoryFormat_g
    )
