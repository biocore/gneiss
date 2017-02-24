# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the GPLv3 License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

import os
import unittest
import subprocess

import numpy as np
import pandas as pd
import pandas.util.testing as pdt

from skbio.stats.composition import ilr_inv
from skbio import TreeNode
import statsmodels.formula.api as smf
import numpy.testing as npt

from skbio.util import get_data_path
from gneiss.regression import mixedlm

from gneiss.regression._regression import lme_regression, ols_regression
from gneiss.regression.tests.test_ols import TestOLS
from gneiss.regression.tests.test_mixedlm import TestMixedLM
from qiime2.metadata import Metadata


class TestOLSPlugin(TestOLS):

    def test_ols_regression(self):
        m = Metadata(self.metadata)
        res = ols_regression(self.table, self.tree, m, 'real')
        res_coef = res.coefficients()
        exp_coef = pd.DataFrame(
            {'Intercept': [0, 1.00],
             'real': [1.0, 0]},
            index=['Y1', 'Y2'])

        pdt.assert_frame_equal(res_coef, exp_coef,
                               check_exact=False,
                               check_less_precise=True)
        # Double check to make sure the fit is perfect
        self.assertAlmostEqual(res.r2, 1)

        # Double check to make sure residuals are zero
        exp_resid = pd.DataFrame([[0., 0.],
                                  [0., 0.],
                                  [0., 0.],
                                  [0., 0.],
                                  [0., 0.]],
                                 index=['s1', 's2', 's3', 's4', 's5'],
                                 columns=['Y1', 'Y2'])
        pdt.assert_frame_equal(exp_resid, res.residuals())

    def test_ols_cli(self):
        # TODO: Is there a better way to test this?
        cmd = ("qiime gneiss ols-regression "
               "--i-table data/test_ols_composition.qza "
               "--i-tree data/test_ols_tree.qza "
               "--o-linear-model test_ols "
               "--p-formula 'ph' "
               "--m-metadata-file data/test_ols_metadata.txt")
        proc = subprocess.Popen(cmd, shell=True)
        proc.wait()
        self.assertTrue(os.path.exists("test_ols.qza"))
        os.remove("test_ols.qza")


class TestMixedLMPlugin(TestMixedLM):

    def test_mixedlm_balances(self):

        res = lme_regression(formula="x1 + x2", table=self.table,
                             metadata=Metadata(self.metadata), tree=self.tree,
                             groups="groups")
        res.fit()
        exp_pvalues = pd.DataFrame(
            [[4.82688604e-236,  4.4193804e-05,  3.972325e-35,  3.568599e-30],
             [0.0994110906314,  4.4193804e-05,  3.972325e-35,  3.568599e-30]],
            index=['Y1', 'Y2'],
            columns=['Intercept', 'groups RE', 'x1', 'x2'])

        pdt.assert_frame_equal(res.pvalues, exp_pvalues,
                               check_less_precise=True)

        exp_coefficients = pd.DataFrame(
            [[4.211451,  0.0935786, 1.022008, 0.924873],
             [0.211451,  0.0935786, 1.022008, 0.924873]],
            columns=['Intercept', 'groups RE', 'x1', 'x2'],
            index=['Y1', 'Y2'])

        pdt.assert_frame_equal(res.coefficients(), exp_coefficients,
                               check_less_precise=True)

    def test_lme_cli(self):
        # TODO: Is there a better way to test this?
        cmd = ("qiime gneiss ols-regression "
               "--i-table data/test_lme_composition.qza "
               "--i-tree data/test_lme_tree.qza "
               "--o-linear-model test_lme "
               "--p-formula 'ph' "
               "--p-groups 'host_subject_id'"
               "--m-metadata-file data/test_lme_metadata.txt")
        proc = subprocess.Popen(cmd, shell=True)
        proc.wait()
        self.assertTrue(os.path.exists("test_lme.qza"))
        os.remove("test_lme.qza")


if __name__ == '__main__':
    unittest.main()
