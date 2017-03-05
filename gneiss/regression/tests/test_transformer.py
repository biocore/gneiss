# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------
import unittest

from gneiss.regression._format import (LinearRegressionFormat_g,
                                       LinearMixedEffectsFormat_g)
from qiime2.plugin.testing import TestPluginBase
from gneiss.regression._ols import OLSModel
from gneiss.regression._mixedlm import LMEModel
import pandas.util.testing as pdt


class TestLinearRegressionTransformers(TestPluginBase):
    package = "gneiss.regression.tests"

    def test_ols_model_to_regression_format(self):

        filepath = self.get_data_path('ols.pickle')
        transformer = self.get_transformer(OLSModel, LinearRegressionFormat_g)
        input = OLSModel.read_pickle(filepath)

        obs = transformer(input)
        obs = OLSModel.read_pickle(str(obs))
        pdt.assert_frame_equal(input.pvalues, obs.pvalues)

    def test_regression_format_to_ols_model(self):
        filename = 'ols.pickle'
        input, obs = self.transform_format(LinearRegressionFormat_g, OLSModel,
                                           filename)

        exp = OLSModel.read_pickle(str(input))
        pdt.assert_frame_equal(exp.pvalues, obs.pvalues)


class TestLinearMixedEffectsTransformers(TestPluginBase):
    package = "gneiss.regression.tests"

    def test_lme_model_to_regression_format(self):
        filepath = self.get_data_path('lme.pickle')
        transformer = self.get_transformer(LMEModel,
                                           LinearMixedEffectsFormat_g)
        input = LMEModel.read_pickle(filepath)

        obs = transformer(input)
        obs = LMEModel.read_pickle(str(obs))
        pdt.assert_frame_equal(input.pvalues, obs.pvalues)

    def test_regression_format_to_lme_model(self):
        filename = 'lme.pickle'
        input, obs = self.transform_format(LinearMixedEffectsFormat_g,
                                           LMEModel, filename)

        exp = LMEModel.read_pickle(str(input))
        pdt.assert_frame_equal(exp.pvalues, obs.pvalues)


if __name__ == '__main__':
    unittest.main()
