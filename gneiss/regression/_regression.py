# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the GPLv3 License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------
import numpy as np
import pandas as pd
from skbio.stats.composition import ilr
from gneiss.util import match, match_tips, rename_internal_nodes
from gneiss.balances import balance_basis

try:
    from ._mixedlm import LinearMixedEffects_g
    from ._ols import Linear_g
    from ._model import RegressionDirectory_g, RegressionDirectoryFormat_g
    from qiime2.plugin import SemanticType
except ImportError:
    raise ImportWarning('Qiime2 not installed.')


plugin.register_formats(RegressionFormat_g, RegressionDirectoryFormat_g)
plugin.register_semantic_type_to_format(
    Regression_g[Linear_g | LinearMixedEffects_g],
    artifact_format=RegressionDirectoryFormat_g
)
