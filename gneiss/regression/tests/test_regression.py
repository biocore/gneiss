# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the GPLv3 License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------
import unittest
from gneiss.util import HAVE_Q2
if HAVE_Q2:
    from gneiss.regression._mixedlm import LinearMixedEffects_g
    from gneiss.regression._ols import Linear_g
    from gneiss.regression._model import (Regression_g,
                                          RegressionFormat_g,
                                          RegressionDirectoryFormat_g)
    from qiime2.plugin import SemanticType
    from qiime2.plugin.testing import TestPluginBase


class TestTypes(TestPluginBase):

    package = "gneiss"

    def test_regression_type_registration(self):
        self.assertRegisteredSemanticType(Regression_g)

    def test_regression_type_registration(self):
        self.assertRegisteredSemanticType(Linear_g)

    def test_regression_type_registration(self):
        self.assertRegisteredSemanticType(LinearMixedEffects_g)

    def test_regression_linear_lme_to_regression_fmt_registration(self):
        self.assertSemanticTypeRegisteredToFormat(
            Regression_g[Linear_g | LinearMixedEffects_g],
            RegressionDirectoryFormat_g)


if __name__ == '__main__':
    unittest.main()
