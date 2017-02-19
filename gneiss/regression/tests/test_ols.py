# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the GPLv3 License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------
import unittest
import numpy as np
import pandas as pd
import pandas.util.testing as pdt
from skbio.stats.composition import ilr_inv
from skbio import TreeNode
from skbio.util import get_data_path
from gneiss.regression import ols


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

        np.random.seed(0)
        n = 15
        a = np.array([1, 4.2, 5.3, -2.2, 8])
        x1 = np.linspace(.01, 0.1, n)
        x2 = np.logspace(0, 0.01, n)
        x3 = np.exp(np.linspace(0, 0.01, n))
        x4 = x1 ** 2
        self.x = pd.DataFrame({'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4})
        y = (a[0] + a[1]*x1 + a[2]*x2 + a[3]*x3 + a[4]*x4 +
             np.random.normal(size=n))
        sy = np.vstack((y, y/10)).T
        self.y = pd.DataFrame(ilr_inv(sy), columns=['a', 'b', 'c'])
        self.t2 = TreeNode.read([r"((a,b)n,c);"])

    def test_ols(self):
        res = ols('real', self.table, self.metadata, self.tree)
        res.fit()
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
        res.fit()
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

        tree = TreeNode.read(['((c,d),(b,a));'])
        exp_tree = TreeNode.read(['((b,a)y1,c)y0;\n'])
        metadata = pd.DataFrame({
            'lame': [1, 1, 1, 1, 1],
            'real': [1, 2, 3, 4, 5]
        }, index=['s1', 's2', 's3', 's4', 's5'])

        res = ols('real + lame', table, metadata, tree)
        res.fit()
        self.assertEqual(str(table), str(exp_table))
        self.assertEqual(str(exp_tree), str(res.tree))

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
            res = ols('real + lame', table, metadata, tree)
            res.fit()

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
            res = ols('real + lame', table, metadata, tree)
            res.fit()

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
            res = ols('real + lame', table, metadata, tree)
            res.fit()

    def test_summary(self):
        A = np.array  # aliasing for the sake of pep8
        table = pd.DataFrame({
            's1': ilr_inv(A([1., 3.])),
            's2': ilr_inv(A([2., 2.])),
            's3': ilr_inv(A([1., 3.])),
            's4': ilr_inv(A([3., 4.])),
            's5': ilr_inv(A([1., 5.]))},
            index=['a', 'b', 'c']).T
        tree = TreeNode.read(['(c, (b,a)Y2)Y1;'])
        metadata = pd.DataFrame({
            'lame': [1, 2, 1, 4, 1],
            'real': [1, 2, 3, 4, 5]
        }, index=['s1', 's2', 's3', 's4', 's5'])

        np.random.seed(0)
        self.maxDiff = None
        model = ols('real', table, metadata, tree)
        model.fit()

        fname = get_data_path('exp_ols_results.txt')
        res = str(model.summary())
        with open(fname, 'r') as fh:
            exp = fh.read()
            self.assertEqual(res, exp)

    def test_summary_head(self):
        A = np.array  # aliasing for the sake of pep8
        table = pd.DataFrame({
            's1': ilr_inv(A([1., 3.])),
            's2': ilr_inv(A([2., 2.])),
            's3': ilr_inv(A([1., 3.])),
            's4': ilr_inv(A([3., 4.])),
            's5': ilr_inv(A([1., 5.]))},
            index=['a', 'b', 'c']).T
        tree = TreeNode.read(['(c, (b,a)Y2)Y1;'])
        metadata = pd.DataFrame({
            'lame': [1, 2, 1, 4, 1],
            'real': [1, 2, 3, 4, 5]
        }, index=['s1', 's2', 's3', 's4', 's5'])

        np.random.seed(0)
        self.maxDiff = None
        model = ols('real', table, metadata, tree)
        model.fit()

        fname = get_data_path('exp_ols_results2.txt')
        res = str(model.summary(ndim=1))
        with open(fname, 'r') as fh:
            exp = fh.read()
            self.assertEqual(res, exp)

    def test_loo(self):
        res = ols(formula="x1 + x2 + x3 + x4",
                  table=self.y, metadata=self.x, tree=self.t2)
        res.fit()
        exp_loo = pd.DataFrame([[0.66953263510975791, 10.994700550912553],
                                [0.69679777354984163, 2.3613911713947062],
                                [0.84934173316473072, 0.4057812892157881],
                                [0.6990546679957772, 2.2872776593899351],
                                [0.72855466737125463, 1.7615637744849277],
                                [0.55998953661859308, 3.617823652256889],
                                [0.81787392852582308, 0.72395497360494043],
                                [0.8653549732546999, 0.17706927499520822],
                                [0.86983181933002329, 0.1216027316667969],
                                [0.87779006612352628, 0.028600627330344405],
                                [0.86591226075609384, 0.16724511075065476],
                                [0.7787232221539, 1.2820054843437292],
                                [0.88032413856094505, 3.4113910096200831e-06],
                                [0.83195133809800792, 0.62276589277034022],
                                [0.85352707356786695, 1.4038585971691198]],
                               columns=['mse', 'pred_err'],
                               index=self.y.index)
        res_loo = res.loo().astype(np.float)
        pdt.assert_frame_equal(exp_loo, res_loo, check_less_precise=True)

    def test_lovo(self):
        res = ols(formula="x1 + x2 + x3 + x4",
                  table=self.y, metadata=self.x, tree=self.t2)
        res.fit()
        exp_lovo = pd.DataFrame([[0.799364, 0.978214],
                                 [0.799363, 0.097355],
                                 [0.799368, 0.0973498],
                                 [0.799364, 0.097354],
                                 [0.799361, 0.0973575]],
                                columns=['mse', 'Rsquared'],
                                index=['Intercept', 'x1', 'x2', 'x3', 'x4'])
        res_lovo = res.lovo().astype(np.float)
        pdt.assert_frame_equal(exp_lovo, res_lovo, check_less_precise=True)

    def test_percent_explained(self):
        res = ols(formula="x1 + x2 + x3 + x4",
                  table=self.y, metadata=self.x, tree=self.t2)
        res.fit()
        res_perc = res.percent_explained()
        exp_perc = pd.Series({'y0': 0.009901,
                              'y1': 0.990099})
        pdt.assert_series_equal(res_perc, exp_perc)

    def test_mse(self):
        res = ols(formula="x1 + x2 + x3 + x4",
                  table=self.y, metadata=self.x, tree=self.t2)
        res.fit()
        self.assertAlmostEqual(res.mse, 0.79228890379010453, places=4)

if __name__ == "__main__":
    unittest.main()
