# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the GPLv3 License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------
import unittest

import os
import subprocess
from gneiss.regression._format import RegressionFormat_g
from qiime2.plugin.testing import TestPluginBase
from gneiss.regression._ols import OLSModel
from gneiss.regression._mixedlm import LMEModel
import pandas.util.testing as pdt


class TestTransformers(TestPluginBase):
    package = "gneiss.regression.tests"

    def test_ols_model_to_regression_format(self):
        # setup
        ols_f = self.get_data_path('ols.pickle.tgz')
        proc = subprocess.Popen('tar -zxvf %s' % ols_f, shell=True)
        proc.wait()

        filepath = 'ols.pickle'
        transformer = self.get_transformer(OLSModel, RegressionFormat_g)
        input = OLSModel.read_pickle(filepath)

        obs = transformer(input)
        obs = OLSModel.read_pickle(str(obs))
        pdt.assert_frame_equal(input.pvalues, obs.pvalues)

    def test_regression_format_to_ols_model(self):
        # setup
        ols_f = self.get_data_path('ols.pickle.tgz')
        proc = subprocess.Popen('tar -zxvf %s' % ols_f, shell=True)
        proc.wait()
        os.rename("ols.pickle", os.path.splitext(ols_f)[0])

        filename = 'ols.pickle'
        input, obs = self.transform_format(RegressionFormat_g, OLSModel,
                                           filename)

        exp = OLSModel.read_pickle(str(input))
        pdt.assert_frame_equal(exp.pvalues, obs.pvalues)

    def test_lme_model_to_regression_format(self):
        # setup
        lme_f = self.get_data_path('lme.pickle.tgz')
        proc = subprocess.Popen('tar -zxvf %s' % lme_f, shell=True)
        proc.wait()

        filepath = 'lme.pickle'
        transformer = self.get_transformer(LMEModel, RegressionFormat_g)
        input = LMEModel.read_pickle(filepath)

        obs = transformer(input)
        obs = LMEModel.read_pickle(str(obs))
        pdt.assert_frame_equal(input.pvalues, obs.pvalues)

    def test_regression_format_to_lme_model(self):
        # setup
        lme_f = self.get_data_path('lme.pickle.tgz')
        proc = subprocess.Popen('tar -zxvf %s' % lme_f, shell=True)
        proc.wait()
        os.rename("lme.pickle", os.path.splitext(lme_f)[0])

        filename = 'lme.pickle'
        input, obs = self.transform_format(RegressionFormat_g, LMEModel,
                                           filename)

        exp = LMEModel.read_pickle(str(input))
        pdt.assert_frame_equal(exp.pvalues, obs.pvalues)


if __name__ == '__main__':
    unittest.main()
