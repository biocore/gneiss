# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the GPLv3 License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------
import numpy as np
import pandas as pd
import pandas.util.testing as pdt
import unittest
from skbio.stats.composition import ilr_inv
from skbio import TreeNode
from gneiss._formula import ols, mixedlm, gee

import numpy.testing as npt

import statsmodels.formula.api as smf
from statsmodels.compat import lrange
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.cov_struct import (Exchangeable, Independence,
                                           GlobalOddsRatio, Autoregressive)
from statsmodels.genmod.families import Gaussian, Binomial, Poisson


class TestOLS(unittest.TestCase):

    def setUp(self):
        A = np.array  # aliasing for the sake of pep8
        self.table = pd.DataFrame({
            's1': ilr_inv(A([1., 1.])),
            's2': ilr_inv(A([1., 2.])),
            's3': ilr_inv(A([1., 3.])),
            's4': ilr_inv(A([1., 4.])),
            's5': ilr_inv(A([1., 5.]))},
            index=['a', 'b', 'c']).T
        self.tree = TreeNode.read(['(c, (b,a)Y2)Y1;'])
        self.unannotated_tree = TreeNode.read(['(c, (b,a));'])
        self.metadata = pd.DataFrame({
            'lame': [1, 1, 1, 1, 1],
            'real': [1, 2, 3, 4, 5]
        }, index=['s1', 's2', 's3', 's4', 's5'])

    def test_ols(self):
        res = ols('real', self.table, self.metadata, self.tree)
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

    def test_ols_rename(self):
        res = ols('real', self.table, self.metadata,
                  self.unannotated_tree)
        res_coef = res.coefficients()
        exp_coef = pd.DataFrame(
            {'Intercept': [0, 1.00],
             'real': [1.0, 0]},
            index=['y0', 'y1'])

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
                                 columns=['y0', 'y1'])
        pdt.assert_frame_equal(exp_resid, res.residuals())

    def test_ols_immutable(self):
        A = np.array  # aliasing for the sake of pep8
        table = pd.DataFrame({
            's1': ilr_inv(A([1., 1.])),
            's2': ilr_inv(A([1., 2.])),
            's3': ilr_inv(A([1., 3.])),
            's4': ilr_inv(A([1., 4.])),
            's5': ilr_inv(A([1., 5.])),
            's6': ilr_inv(A([1., 5.]))},
            index=['a', 'b', 'c']).T
        exp_table = pd.DataFrame({
            's1': ilr_inv(A([1., 1.])),
            's2': ilr_inv(A([1., 2.])),
            's3': ilr_inv(A([1., 3.])),
            's4': ilr_inv(A([1., 4.])),
            's5': ilr_inv(A([1., 5.])),
            's6': ilr_inv(A([1., 5.]))},
            index=['a', 'b', 'c']).T

        tree = TreeNode.read(['((c,d),(b,a)Y2)Y1;'])
        exp_tree = TreeNode.read(['((c,d),(b,a)Y2)Y1;'])
        metadata = pd.DataFrame({
            'lame': [1, 1, 1, 1, 1],
            'real': [1, 2, 3, 4, 5]
        }, index=['s1', 's2', 's3', 's4', 's5'])

        ols('real + lame', table, metadata, tree)
        self.assertEqual(str(table), str(exp_table))
        self.assertEqual(str(exp_tree), str(tree))

    def test_ols_empty_table_error(self):
        A = np.array  # aliasing for the sake of pep8
        table = pd.DataFrame({
            's1': ilr_inv(A([1., 1.])),
            's2': ilr_inv(A([1., 2.])),
            's3': ilr_inv(A([1., 3.])),
            's4': ilr_inv(A([1., 4.])),
            's5': ilr_inv(A([1., 5.])),
            's6': ilr_inv(A([1., 5.]))},
            index=['x', 'y', 'z']).T

        tree = TreeNode.read(['((c,d),(b,a)Y2)Y1;'])
        metadata = pd.DataFrame({
            'lame': [1, 1, 1, 1, 1],
            'real': [1, 2, 3, 4, 5]
        }, index=['s1', 's2', 's3', 's4', 's5'])
        with self.assertRaises(ValueError):
            ols('real + lame', table, metadata, tree)

    def test_ols_empty_metadata_error(self):
        A = np.array  # aliasing for the sake of pep8
        table = pd.DataFrame({
            'k1': ilr_inv(A([1., 1.])),
            'k2': ilr_inv(A([1., 2.])),
            'k3': ilr_inv(A([1., 3.])),
            'k4': ilr_inv(A([1., 4.])),
            'k5': ilr_inv(A([1., 5.])),
            'k6': ilr_inv(A([1., 5.]))},
            index=['a', 'b', 'c']).T

        tree = TreeNode.read(['((c,d),(b,a)Y2)Y1;'])
        metadata = pd.DataFrame({
            'lame': [1, 1, 1, 1, 1],
            'real': [1, 2, 3, 4, 5]
        }, index=['s1', 's2', 's3', 's4', 's5'])
        with self.assertRaises(ValueError):
            ols('real + lame', table, metadata, tree)

    def test_ols_zero_error(self):
        table = pd.DataFrame({
            's1': [0, 0, 0],
            's2': [0, 0, 0],
            's3': [0, 0, 0],
            's4': [0, 0, 0],
            's5': [0, 0, 0],
            's6': [0, 0, 0]},
            index=['a', 'b', 'c']).T

        tree = TreeNode.read(['((c,d),(b,a)Y2)Y1;'])
        metadata = pd.DataFrame({
            'lame': [1, 1, 1, 1, 1],
            'real': [1, 2, 3, 4, 5]
        }, index=['s1', 's2', 's3', 's4', 's5'])
        with self.assertRaises(ValueError):
            ols('real + lame', table, metadata, tree)


class TestMixedLM(unittest.TestCase):

    def setUp(self):
        A = np.array  # aliasing for the sake of pep8
        self.table = pd.DataFrame({
            'x1': ilr_inv(A([1., 1.])),
            'x2': ilr_inv(A([1., 2.])),
            'x3': ilr_inv(A([1., 3.])),
            'y1': ilr_inv(A([1., 2.])),
            'y2': ilr_inv(A([1., 3.])),
            'y3': ilr_inv(A([1., 4.])),
            'z1': ilr_inv(A([1., 5.])),
            'z2': ilr_inv(A([1., 6.])),
            'z3': ilr_inv(A([1., 7.])),
            'u1': ilr_inv(A([1., 6.])),
            'u2': ilr_inv(A([1., 7.])),
            'u3': ilr_inv(A([1., 8.]))},
            index=['a', 'b', 'c']).T
        self.tree = TreeNode.read(['(c, (b,a)Y2)Y1;'])
        self.metadata = pd.DataFrame({
            'patient': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
            'treatment': [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
            'time': [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]
        }, index=['x1', 'x2', 'x3', 'y1', 'y2', 'y3',
                  'z1', 'z2', 'z3', 'u1', 'u2', 'u3'])

    # test case borrowed from statsmodels
    # https://github.com/statsmodels/statsmodels/blob/master/statsmodels
    # /regression/tests/test_lme.py#L254
    def test_mixedlm_univariate(self):

        np.random.seed(6241)
        n = 1600
        exog = np.random.normal(size=(n, 2))
        groups = np.kron(np.arange(n / 16), np.ones(16))

        # Build up the random error vector
        errors = 0

        # The random effects
        exog_re = np.random.normal(size=(n, 2))
        slopes = np.random.normal(size=(n / 16, 2))
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
        df["y"] = endog
        df["groups"] = groups
        df["x1"] = exog[:, 0]
        df["x2"] = exog[:, 1]

        # Equivalent model in R:
        # df.to_csv("tst.csv")
        # model = lmer(y ~ x1 + x2 + (0 + z1 + z2 | groups) + (1 | v1) + (1 |
        # v2), df)

        model1 = smf.mixedlm("y ~ x1 + x2", groups=groups,
                             data=df)
        result1 = model1.fit()

        # Compare to R
        npt.assert_allclose(result1.fe_params, [
            0.211451, 1.022008, 0.924873], rtol=1e-2)
        npt.assert_allclose(result1.cov_re, [[0.987738]], rtol=1e-3)

        npt.assert_allclose(result1.bse.iloc[0:3], [
            0.128377,  0.082644,  0.081031], rtol=1e-3)

    def test_mixedlm_balances(self):
        np.random.seed(6241)
        n = 1600
        exog = np.random.normal(size=(n, 2))
        groups = np.kron(np.arange(n / 16), np.ones(16))

        # Build up the random error vector
        errors = 0

        # The random effects
        exog_re = np.random.normal(size=(n, 2))
        slopes = np.random.normal(size=(n / 16, 2))
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

        tree = TreeNode.read(['(c, (b,a)Y2)Y1;'])
        iv = ilr_inv(df[["y1", "y2"]].values)
        table = pd.DataFrame(iv, columns=['a', 'b', 'c'])
        metadata = df[['x1', 'x2', 'groups']]

        res = mixedlm("x1 + x2", table, metadata, tree, groups="groups")
        exp_pvalues = pd.DataFrame(
            [[4.923122e-236,  3.180390e-40,  3.972325e-35,  3.568599e-30],
             [9.953418e-02,  3.180390e-40,  3.972325e-35,  3.568599e-30]],
            index=['Y1', 'Y2'],
            columns=['Intercept', 'Intercept RE', 'x1', 'x2'])

        pdt.assert_frame_equal(res.pvalues, exp_pvalues,
                               check_less_precise=True)

        exp_coefficients = pd.DataFrame(
            [[4.211451,  -0.305906, 1.022008, 0.924873],
             [0.211451,  -0.305906, 1.022008, 0.924873]],
            columns=['Intercept', 'Intercept RE', 'x1', 'x2'],
            index=['Y1', 'Y2'])

        pdt.assert_frame_equal(res.coefficients(), exp_coefficients,
                               check_less_precise=True)

    def test_mixedlm_balances_vcf(self):
        np.random.seed(6241)
        n = 1600
        exog = np.random.normal(size=(n, 2))
        groups = np.kron(np.arange(n / 16), np.ones(16))

        # Build up the random error vector
        errors = 0

        # The random effects
        exog_re = np.random.normal(size=(n, 2))
        slopes = np.random.normal(size=(n / 16, 2))
        slopes = np.kron(slopes, np.ones((16, 1))) * exog_re
        errors += slopes.sum(1)

        # First variance component
        subgroups1 = np.kron(np.arange(n / 4), np.ones(4))
        errors += np.kron(2 * np.random.normal(size=n // 4), np.ones(4))

        # Second variance component
        subgroups2 = np.kron(np.arange(n / 2), np.ones(2))
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

        tree = TreeNode.read(['(c, (b,a)Y2)Y1;'])
        iv = ilr_inv(df[["y1", "y2"]].values)
        table = pd.DataFrame(iv, columns=['a', 'b', 'c'])
        metadata = df[['x1', 'x2', 'z1', 'z2', 'v1', 'v2', 'groups']]

        res = mixedlm("x1 + x2", table, metadata, tree, groups="groups",
                      re_formula="0+z1+z2")

        exp_pvalues = pd.DataFrame(
            [[4.923122e-236,  3.180390e-40,  3.972325e-35,  3.568599e-30],
             [9.953418e-02,  3.180390e-40,  3.972325e-35,  3.568599e-30]],
            index=['Y1', 'Y2'],
            columns=['Intercept', 'Intercept RE', 'x1', 'x2'])

        exp_pvalues = pd.DataFrame([
            [0.000000, 3.858750e-39, 2.245068e-33,
             2.434437e-35, 0.776775, 6.645741e-34],
            [0.038015, 3.858750e-39, 2.245068e-33,
             2.434437e-35, 0.776775, 6.645741e-34]],
            columns=['Intercept', 'x1', 'x2', 'z1 RE',
                     'z1 RE x z2 RE', 'z2 RE'],
            index=['Y1', 'Y2'])
        exp_coefficients = pd.DataFrame(
            [[4.163141, 1.030013, 0.935514, 0.339239, -0.005792, 0.38456],
             [0.163141, 1.030013, 0.935514, 0.339239, -0.005792, 0.38456]],
            columns=['Intercept', 'x1', 'x2', 'z1 RE',
                     'z1 RE x z2 RE', 'z2 RE'],
            index=['Y1', 'Y2'])

        pdt.assert_frame_equal(res.pvalues, exp_pvalues,
                               check_less_precise=True)

        pdt.assert_frame_equal(res.coefficients(), exp_coefficients,
                               check_less_precise=True)

    def test_mixedlm_zero_error(self):
        table = pd.DataFrame({
            's1': [0, 0, 0],
            's2': [0, 0, 0],
            's3': [0, 0, 0],
            's4': [0, 0, 0],
            's5': [0, 0, 0],
            's6': [0, 0, 0]},
            index=['a', 'b', 'c']).T

        tree = TreeNode.read(['((c,d),(b,a)Y2)Y1;'])
        metadata = pd.DataFrame({
            'lame': [1, 1, 1, 2, 2],
            'real': [1, 2, 3, 4, 5]
        }, index=['s1', 's2', 's3', 's4', 's5'])
        with self.assertRaises(ValueError):
            mixedlm('real + lame', table, metadata, tree, groups='lame')


class TestGEE(unittest.TestCase):

    def test_formulas(self):
        # Obtained from statsmodels tests for gee
        np.random.seed(0)
        n = 100
        y = np.random.normal(size=n)
        y_ = y.reshape((len(y), 1))
        y_ = np.hstack((y_*2, y_))

        tree = TreeNode.read(['(c, (b,a)Y2)Y1;'])

        Y = ilr_inv(y_)

        X1 = np.random.normal(size=n)
        mat = np.concatenate((np.ones((n, 1)), X1[:, None]), axis=1)
        Time = np.random.uniform(size=n)
        groups = np.kron(lrange(20), np.ones(5))

        table = pd.DataFrame(Y, columns=['a', 'b', 'c'])
        from gneiss.util import match_tips
        xt, xr = match_tips(table, tree)

        metadata = pd.DataFrame({"X1": X1, "Time": Time, "groups": groups})

        va = Autoregressive()
        family = Gaussian()

        mod1 = GEE(y, mat, groups, time=Time, family=family,
                   cov_struct=va)
        rslt1 = mod1.fit()

        mod2 = gee("X1", table, metadata, tree,
                   groups=groups, time=Time, family=family,
                   cov_struct=va)
        rslt2 = mod2.results[0]

        mod3 = gee("X1", table, metadata, tree,
                   groups=groups, time="Time", family=family,
                   cov_struct=va)
        rslt3 = mod3.results[0]

        mod4 = gee("X1", table, metadata, tree,
                   groups="groups", time=Time,
                   family=family, cov_struct=va)
        rslt4 = mod4.results[0]

        mod5 = gee("X1", table, metadata, tree,
                   groups="groups", time="Time",
                   family=family, cov_struct=va)
        rslt5 = mod5.results[0]

        npt.assert_almost_equal(rslt1.params, rslt2.params, decimal=8)
        npt.assert_almost_equal(rslt1.params, rslt3.params, decimal=8)
        npt.assert_almost_equal(rslt1.params, rslt4.params, decimal=8)
        npt.assert_almost_equal(rslt1.params, rslt5.params, decimal=8)


if __name__ == '__main__':
    unittest.main()

