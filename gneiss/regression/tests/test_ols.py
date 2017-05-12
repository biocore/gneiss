# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------
import os
import shutil
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

        # for testing the plugins
        self.results = "results"
        os.mkdir(self.results)

        self.table2 = pd.DataFrame({
            's1': A([1., 1.]),
            's2': A([1., 2.]),
            's3': A([1., 3.]),
            's4': A([1., 4.]),
            's5': A([1., 5.]),
            's6': A([1., 6.]),
            's7': A([1., 7.]),
            's8': A([1., 8.]),
            's9': A([1., 9.]),
            's10': A([1., 10.]),
            's11': A([1., 11.]),
            's12': A([1., 12.]),
            's13': A([1., 13.]),
            's14': A([1., 14.]),
            's15': A([1., 15.])},
            index=['Y1', 'Y2']).T

        self.metadata2 = pd.DataFrame({
            'lame': [1, 1, 1, 1, 1,
                     1, 1, 1, 1, 1,
                     1, 1, 1, 1, 1],
            'real': [1, 2, 3, 4, 5,
                     6, 7, 8, 9, 10,
                     11, 12, 13, 14, 15]
        }, index=['s1', 's2', 's3', 's4', 's5',
                  's6', 's7', 's8', 's9', 's10',
                  's11', 's12', 's13', 's14', 's15'])


    def tearDown(self):
        shutil.rmtree(self.results)


class TestOLSFunctions(TestOLS):

    def test_ols(self):
        res = ols('real', self.table, self.metadata)
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

        res = ols('real + lame', table, metadata)
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

        res = ols('real + lame', table, metadata)
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
        res = ols('real + lame', table, metadata)
        res.fit()

        exp_coefs = pd.DataFrame(
            [[-7.494005e-16, -7.494005e-16, 1.000000e+00],
             [5.000000e-01, 5.000000e-01, -1.294503e-16]],
            columns=['Intercept', 'lame', 'real'], index=['y0', 'y1'])
        res_coefs = res.coefficients().sort_index()

        pdt.assert_frame_equal(exp_coefs, res_coefs,
                               check_less_precise=True)

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
            res = ols('real + lame', table, metadata)
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
        metadata = pd.DataFrame({
            'lame': [1, 2, 1, 4, 1],
            'real': [1, 2, 3, 4, 5]
        }, index=['s1', 's2', 's3', 's4', 's5'])

        np.random.seed(0)
        self.maxDiff = None
        model = ols('real', table, metadata)
        model.fit()
        _l = model.lovo()
        _k = model.kfold(num_folds=2)
        fname = get_data_path('exp_ols_results.txt')
        res = str(model.summary(_k, _l))
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
                  table=y, metadata=x)
        res.fit()
        res_cv = res.kfold(num_folds=2)
        exp_cv = pd.DataFrame({
            'fold_0': [0.000353, 0.786741, 332.452084],
            'fold_1': [0.000043, 0.838284, 0.230325]},
            index=['model_mse', 'Rsquared', 'pred_mse']).T
        # Precision issues ...
        # pdt.assert_frame_equal(res_cv, exp_cv, check_less_precise=True)
        npt.assert_allclose(exp_cv, res_cv, atol=1e-3,  rtol=1e-3)

    def test_loo(self):
        res = ols(formula="x1 + x2 + x3 + x4",
                  table=self.y, metadata=self.x)
        res.fit()
        exp_loo = pd.DataFrame([[0.001479, 4.583677e-04],
                                [0.001426, 1.893037e-04],
                                [0.001558, 2.372980e-07],
                                [0.001401, 2.204438e-04],
                                [0.001339, 2.830608e-04],
                                [0.000424, 1.422739e-03],
                                [0.001353, 2.647397e-04],
                                [0.001555, 4.301729e-06],
                                [0.001555, 3.513098e-06],
                                [0.001538, 2.457823e-05],
                                [0.001549, 1.178413e-05],
                                [0.001517, 5.696703e-05],
                                [0.001557, 1.387215e-06],
                                [0.001502, 7.969229e-05],
                                [0.001497, 3.562734e-04]],
                               columns=['mse', 'pred_err'],
                               index=self.y.index)

        res_loo = res.loo()
        # Precision issues ...
        # pdt.assert_frame_equal(exp_loo, res_loo, check_less_precise=True)
        npt.assert_allclose(exp_loo, res_loo, atol=1e-3,  rtol=1e-3)

    def test_lovo(self):
        res = ols(formula="x1 + x2 + x3 + x4",
                  table=self.y, metadata=self.x)
        res.fit()

        exp_lovo = pd.DataFrame([[0.000458, 0.999651, -0.804403],
                                 [0.000458, 0.133113, 0.062136],
                                 [0.000458, 0.133046, 0.062203],
                                 [0.000458, 0.133100, 0.062148],
                                 [0.000458, 0.133143, 0.062106]],
                                columns=['mse', 'Rsquared', 'R2diff'],
                                index=['Intercept', 'x1', 'x2', 'x3', 'x4'])
        res_lovo = res.lovo()
        # Precision issues ...
        # pdt.assert_frame_equal(exp_lovo, res_lovo, check_less_precise=True)
        npt.assert_allclose(exp_lovo, res_lovo, atol=1e-3,  rtol=1e-3)

    def test_percent_explained(self):
        table = ilr_transform(self.y, self.t2)
        res = ols(formula="x1 + x2 + x3 + x4",
                  table=table, metadata=self.x)
        res.fit()
        res_perc = res.percent_explained()
        exp_perc = pd.Series({'y0': 0.009901,
                              'y1': 0.990099})
        pdt.assert_series_equal(res_perc, exp_perc)

    def test_mse(self):
        table = ilr_transform(self.y, self.t2)
        res = ols(formula="x1 + x2 + x3 + x4",
                  table=table, metadata=self.x)
        res.fit()
        self.assertAlmostEqual(res.mse, 0.79228890379010453, places=4)

    def test_write(self):
        table = ilr_transform(self.y, self.t2)
        res = ols(formula="x1 + x2 + x3 + x4",
                  table=table, metadata=self.x)
        res.fit()
        res.write_pickle('ols.pickle')


if __name__ == "__main__":
    unittest.main()
