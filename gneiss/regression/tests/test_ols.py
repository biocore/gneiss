# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the Modified BSD License.
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
from gneiss.composition import ilr_transform
import numpy.testing as npt


class TestOLS(unittest.TestCase):
    def setUp(self):
        A = np.array  # aliasing for the sake of pep8
        self.table = pd.DataFrame({
            's1': A([1., 1.]),
            's2': A([1., 2.]),
            's3': A([1., 3.]),
            's4': A([1., 4.]),
            's5': A([1., 5.])},
            index=['Y1', 'Y2']).T
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
        self.t2 = TreeNode.read([r"((a,b)y1,c)y0;"])


class TestOLSFunctions(TestOLS):

    def test_ols(self):
        res = ols('real', self.table, self.metadata, self.tree)
        res.fit()
        res_coef = res.coefficients()
        exp_coef = pd.DataFrame(
            {'Intercept': [1.00, 0],
             'real': [0, 1.0]},
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

    def test_ols_table_immutable(self):
        # test to see if values in table get filtered out.
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

        tree = TreeNode.read(['((b,a)y1,c)y0;\n'])
        metadata = pd.DataFrame({
            'lame': [1, 1, 1, 1, 1],
            'real': [1, 2, 3, 4, 5]
        }, index=['s1', 's2', 's3', 's4', 's5'])

        res = ols('real + lame', table, metadata, tree)
        res.fit()
        self.assertEqual(str(table), str(exp_table))

    def test_ols_metadata_immutable(self):
        # test to see if values in table get filtered out.
        A = np.array  # aliasing for the sake of pep8
        table = pd.DataFrame({
            's1': ilr_inv(A([1., 1.])),
            's2': ilr_inv(A([1., 2.])),
            's3': ilr_inv(A([1., 3.])),
            's4': ilr_inv(A([1., 4.])),
            's5': ilr_inv(A([1., 5.]))},
            index=['a', 'b', 'c']).T
        exp_table = pd.DataFrame({
            's1': ilr_inv(A([1., 1.])),
            's2': ilr_inv(A([1., 2.])),
            's3': ilr_inv(A([1., 3.])),
            's4': ilr_inv(A([1., 4.])),
            's5': ilr_inv(A([1., 5.]))},
            index=['a', 'b', 'c']).T

        tree = TreeNode.read(['((b,a)y1,c)y0;\n'])
        metadata = pd.DataFrame({
            'lame': [1, 1, 1, 1, 1, 1],
            'real': [1, 2, 3, 4, 5, 1]
        }, index=['s1', 's2', 's3', 's4', 's5', 's6'])

        exp_metadata = pd.DataFrame({
            'lame': [1, 1, 1, 1, 1, 1],
            'real': [1, 2, 3, 4, 5, 1]
        }, index=['s1', 's2', 's3', 's4', 's5', 's6'])

        res = ols('real + lame', table, metadata, tree)
        res.fit()
        self.assertEqual(str(table), str(exp_table))
        self.assertEqual(str(metadata), str(exp_metadata))

    def test_ols_missing_metadata(self):
        np.random.seed(0)
        A = np.array  # aliasing for the sake of pep8
        table = pd.DataFrame({
            's1': A([1., 1.]),
            's2': A([1., 2.]),
            's3': A([1., 3.]),
            's4': A([1., 4.]),
            's5': A([1., 5.]),
            's6': A([1., 5.]),
            's7': A([1., 5.])},
            index=['y1', 'y0']).T

        tree = TreeNode.read(['(c, (b,a)y1)y0;\n'])
        metadata = pd.DataFrame({
            'lame': [1, 1, 1, 1, 1, 0],
            'real': [1, 2, 3, 4, 5, np.nan]
        }, index=['s1', 's2', 's3', 's4', 's5', 's7'])
        res = ols('real + lame', table, metadata, tree)
        res.fit()

        exp_coefs = pd.DataFrame(
            [[-7.494005e-16, -7.494005e-16, 1.000000e+00],
             [5.000000e-01, 5.000000e-01, -1.294503e-16]],
            columns=['Intercept', 'lame', 'real'], index=['y0', 'y1'])
        res_coefs = res.coefficients().sort_index()

        pdt.assert_frame_equal(exp_coefs, res_coefs,
                               check_less_precise=True)

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
            's1': A([1., 3.]),
            's2': A([2., 2.]),
            's3': A([1., 3.]),
            's4': A([3., 4.]),
            's5': A([1., 5.])},
            index=['Y2', 'Y1']).T
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

    def test_kfold(self):
        np.random.seed(0)
        n = 15
        a = np.array([1, 4.2, 5.3, -2.2, 8])
        x1 = np.linspace(.01, 0.1, n)
        x2 = np.logspace(0, 0.01, n)
        x3 = np.exp(np.linspace(0, 0.01, n))
        x4 = x1 ** 2

        x = pd.DataFrame({'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4})
        y = (a[0] + a[1]*x1 + a[2]*x2 + a[3]*x3 + a[4]*x4 +
             np.random.normal(size=n))
        sy = np.vstack((y, y/10)).T
        y = pd.DataFrame(ilr_inv(sy), columns=['a', 'b', 'c'])
        t2 = TreeNode.read([r"((a,b)n,c)f;"])
        res = ols(formula="x1 + x2 + x3 + x4",
                  table=y, metadata=x, tree=t2)
        res.fit()
        res_cv = res.kfold(num_folds=2)
        exp_cv = pd.DataFrame({
            'fold_0': [0.000530145, 4.10659e-05],
            'fold_1': [6.41364e-05, 0.000622167]},
            index=['model_mse', 'pred_mse']).T
        pdt.assert_frame_equal(res_cv, exp_cv)

    def test_loo(self):
        res = ols(formula="x1 + x2 + x3 + x4",
                  table=self.y, metadata=self.x, tree=self.t2)
        res.fit()
        exp_loo = pd.DataFrame([[0.000493, 1.375103e-03],
                                [0.000475, 5.679110e-04],
                                [0.000519, 7.118939e-07],
                                [0.000467, 6.613313e-04],
                                [0.000446, 8.491825e-04],
                                [0.000141, 4.268216e-03],
                                [0.000451, 7.942190e-04],
                                [0.000518, 1.290519e-05],
                                [0.000518, 1.053929e-05],
                                [0.000513, 7.373469e-05],
                                [0.000516, 3.535239e-05],
                                [0.000506, 1.709011e-04],
                                [0.000519, 4.161646e-06],
                                [0.000501, 2.390769e-04],
                                [0.000499, 1.068820e-03]],
                               columns=['mse', 'pred_err'],
                               index=self.y.index)

        res_loo = res.loo()
        # Precision issues ...
        # pdt.assert_frame_equal(exp_loo, res_loo, check_less_precise=True)
        npt.assert_allclose(exp_loo, res_loo, atol=1e-3,  rtol=1e-3)

    def test_lovo(self):
        res = ols(formula="x1 + x2 + x3 + x4",
                  table=self.y, metadata=self.x, tree=self.t2)
        res.fit()

        exp_lovo = pd.DataFrame([[0.000457738, 0.999651],
                                 [0.000457738, 0.133113],
                                 [0.000457738, 0.133046],
                                 [0.000457738, 0.133100],
                                 [0.000457738, 0.133143]],
                                columns=['mse', 'Rsquared'],
                                index=['Intercept', 'x1', 'x2', 'x3', 'x4'])
        res_lovo = res.lovo()
        pdt.assert_frame_equal(exp_lovo, res_lovo, check_less_precise=True)

    def test_percent_explained(self):
        table = ilr_transform(self.y, self.t2)
        res = ols(formula="x1 + x2 + x3 + x4",
                  table=table, metadata=self.x, tree=self.t2)
        res.fit()
        res_perc = res.percent_explained()
        exp_perc = pd.Series({'y0': 0.009901,
                              'y1': 0.990099})
        pdt.assert_series_equal(res_perc, exp_perc)

    def test_mse(self):
        table = ilr_transform(self.y, self.t2)
        res = ols(formula="x1 + x2 + x3 + x4",
                  table=table, metadata=self.x, tree=self.t2)
        res.fit()
        self.assertAlmostEqual(res.mse, 0.79228890379010453, places=4)

    def test_write(self):
        table = ilr_transform(self.y, self.t2)
        res = ols(formula="x1 + x2 + x3 + x4",
                  table=table, metadata=self.x, tree=self.t2)
        res.fit()
        res.write_pickle('ols.pickle')


if __name__ == "__main__":
    unittest.main()
