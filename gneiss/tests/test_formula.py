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
from gneiss._formula import ols, mixedlm
import statsmodels.formula.api as smf
import numpy.testing as npt


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

    def test_ols_empty_table(self):
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

    def test_ols_empty_metadata(self):
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


if __name__ == '__main__':
    unittest.main()
