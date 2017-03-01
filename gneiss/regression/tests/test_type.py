# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the GPLv3 License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------
import unittest

from gneiss.regression._type import (LinearMixedEffects_g,
                                     LinearRegression_g)
from gneiss.regression._format import (LinearRegressionDirectoryFormat_g,
                                       LinearMixedEffectsDirectoryFormat_g)
from qiime2.plugin.testing import TestPluginBase


class TestLinearType(TestPluginBase):

    package = "gneiss.regression.tests"

    def test_linear_type_registration(self):
        self.assertRegisteredSemanticType(LinearRegression_g)

    def test_regression_linear_lme_to_regression_fmt_registration(self):
        self.assertSemanticTypeRegisteredToFormat(
            LinearRegression_g,
            LinearRegressionDirectoryFormat_g)


class TestLinearMixedEffectsType(TestPluginBase):

    package = "gneiss.regression.tests"

    def test_linear_mixed_effects_type_registration(self):
        self.assertRegisteredSemanticType(LinearMixedEffects_g)

    def test_regression_linear_lme_to_regression_fmt_registration(self):
        self.assertSemanticTypeRegisteredToFormat(
            LinearMixedEffects_g,
            LinearMixedEffectsDirectoryFormat_g)


if __name__ == '__main__':
    unittest.main()
