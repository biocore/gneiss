# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------
import os
import shutil
import numpy as np
import pandas as pd
import pandas.util.testing as pdt
import unittest
from skbio import TreeNode
from gneiss.regression import mixedlm


class TestMixedLM(unittest.TestCase):

    def setUp(self):
        np.random.seed(6241)
        n = 1600
        exog = np.random.normal(size=(n, 2))
        groups = np.kron(np.arange(n // 16), np.ones(16))

        # Build up the random error vector
        errors = 0

        # The random effects
        exog_re = np.random.normal(size=(n, 2))
        slopes = np.random.normal(size=(n // 16, 2))
        slopes = np.kron(slopes, np.ones((16, 1))) * exog_re
        errors += slopes.sum(1)

        # First variance component
        errors += np.kron(2 * np.random.normal(size=n // 4), np.ones(4))

        # Second variance component
        errors += np.kron(2 * np.random.normal(size=n // 2), np.ones(2))

        # iid errors
        errors += np.random.normal(size=n)

        endog = exog.sum(1) + errors

        df = pd.DataFrame(index=range(n))
        df["y1"] = endog
        df["y2"] = endog + 2 * 2
        df["groups"] = groups
        df["x1"] = exog[:, 0]
        df["x2"] = exog[:, 1]

        self.tree = TreeNode.read(['(c, (b,a)y2)y1;'])
        self.table = df[["y1", "y2"]]
        self.metadata = df[['x1', 'x2', 'groups']]

        # for testing the plugins
        self.results = "results"
        if not os.path.exists(self.results):
            os.mkdir(self.results)

    def tearDown(self):
        shutil.rmtree(self.results)


class TestMixedLMFunctions(TestMixedLM):

    def test_mixedlm_balances(self):

        res = mixedlm("x1 + x2", self.table, self.metadata,
                      groups="groups")
        res.fit()
        exp_pvalues = pd.DataFrame(
            [[0.0994110906314, 4.4193804e-05, 3.972325e-35, 3.568599e-30],
             [4.82688604e-236, 4.4193804e-05, 3.972325e-35, 3.568599e-30]],
            index=['y1', 'y2'],
            columns=['Intercept', 'Group Var', 'x1', 'x2']).T

        res_pvals = res.pvalues.sort_index(axis=0).sort_index(axis=1)
        exp_pvals = exp_pvalues.sort_index(axis=0).sort_index(axis=1)

        pdt.assert_frame_equal(res_pvals, exp_pvals,
                               check_less_precise=True)

        exp_coefficients = pd.DataFrame(
            [[0.211451, 0.0935786, 1.022008, 0.924873],
             [4.211451, 0.0935786, 1.022008, 0.924873]],
            columns=['Intercept', 'Group Var', 'x1', 'x2'],
            index=['y1', 'y2']).sort_index().T
        res_coef = res.coefficients().sort_index(axis=0).sort_index(axis=1)
        exp_coef = exp_coefficients.sort_index(axis=0).sort_index(axis=1)

        pdt.assert_frame_equal(res_coef, exp_coef,
                               check_less_precise=True)

    def test_mixedlm_balances_vcf(self):
        np.random.seed(6241)
        n = 1600
        exog = np.random.normal(size=(n, 2))
        groups = np.kron(np.arange(n // 16), np.ones(16))

        # Build up the random error vector
        errors = 0

        # The random effects
        exog_re = np.random.normal(size=(n, 2))
        slopes = np.random.normal(size=(n // 16, 2))
        slopes = np.kron(slopes, np.ones((16, 1))) * exog_re
        errors += slopes.sum(1)

        # First variance component
        subgroups1 = np.kron(np.arange(n // 4), np.ones(4))
        errors += np.kron(2 * np.random.normal(size=n // 4), np.ones(4))

        # Second variance component
        subgroups2 = np.kron(np.arange(n // 2), np.ones(2))
        errors += np.kron(2 * np.random.normal(size=n // 2), np.ones(2))

        # iid errors
        errors += np.random.normal(size=n)

        endog = exog.sum(1) + errors

        df = pd.DataFrame(index=range(n))
        df["y1"] = endog
        df["y2"] = endog + 2 * 2
        df["groups"] = groups
        df["x1"] = exog[:, 0]
        df["x2"] = exog[:, 1]
        df["z1"] = exog_re[:, 0]
        df["z2"] = exog_re[:, 1]
        df["v1"] = subgroups1
        df["v2"] = subgroups2

        table = df[["y1", "y2"]]
        metadata = df[['x1', 'x2', 'z1', 'z2', 'v1', 'v2', 'groups']]

        res = mixedlm("x1 + x2", table, metadata, groups="groups",
                      re_formula="0+z1+z2")
        res.fit()

        exp_pvalues = pd.DataFrame([
            [0.038015, 3.858750e-39, 2.245068e-33,
             2.552217e-05, 0.923418, 6.645741e-34],
            [0.000000, 3.858750e-39, 2.245068e-33,
             2.552217e-05, 0.923418, 6.645741e-34]],
            columns=['Intercept', 'x1', 'x2', 'z1 Var',
                     'z1 x z2 Cov', 'z2 Var'],
            index=['y1', 'y2']).T

        exp_coefficients = pd.DataFrame(
            [[0.163141, 1.030013, 0.935514, 0.115082, -0.001962, 0.14792],
             [4.163141, 1.030013, 0.935514, 0.115082, -0.001962, 0.14792]],
            columns=['Intercept', 'x1', 'x2', 'z1 Var',
                     'z1 x z2 Cov', 'z2 Var'],
            index=['y1', 'y2']).T

        pdt.assert_frame_equal(res.pvalues.sort_index(axis=0),
                               exp_pvalues.sort_index(axis=0),
                               check_less_precise=True)

        pdt.assert_frame_equal(res.coefficients().sort_index(axis=0),
                               exp_coefficients.sort_index(axis=0),
                               check_less_precise=True)

    def test_percent_explained(self):
        model = mixedlm("x1 + x2", self.table, self.metadata,
                        groups="groups")

        model.fit()
        res = model.percent_explained()
        exp = pd.Series([0.5, 0.5], index=['y1', 'y2'])
        pdt.assert_series_equal(res, exp, check_less_precise=True)


if __name__ == '__main__':
    unittest.main()
