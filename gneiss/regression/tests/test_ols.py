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
from gneiss.balances import balance_basis
import statsmodels.formula.api as smf


class TestOLS(unittest.TestCase):
    """ Tests OLS regression with refactored matrix multiplication. """
    def setUp(self):
        np.random.seed(0)
        b01, b11, b21 = 1, 2, -3
        b02, b12, b22 = 2, -1, 4
        n = 50
        x1 = np.linspace(0, 10, n)
        x2 = np.linspace(10, 15, n)
        e = np.random.normal(size=n) * 10
        y1 = b01 + b11 * x1 + b21 * x2 + e
        e = np.random.normal(size=n) * 10
        y2 = b02 + b12 * x1 + b22 * x2 + e
        Y = pd.DataFrame(np.vstack((y1, y2)).T,
                         columns=['y1', 'y2'])

        B = pd.DataFrame([[b01, b11, b21],
                          [b02, b12, b22]])

        X = pd.DataFrame(
            np.vstack((np.ones(n), x1, x2)).T,
            columns=['Intercept', 'x1', 'x2'])

        self.Y = Y
        self.B = B
        self.X = X
        self.r1_ = smf.OLS(endog=y1, exog=X).fit()
        self.r2_ = smf.OLS(endog=y2, exog=X).fit()
        self.tree = TreeNode.read(['(c, (b,a)y2)y1;'])

        self.results = "results"
        if not os.path.exists(self.results):
            os.mkdir(self.results)

    def tearDown(self):
        shutil.rmtree(self.results)

    def test_ols_immutable(self):
        # test to see if values in table get filtered out.
        # and that the original table doesn't change
        table = self.Y
        x = pd.DataFrame(self.X.values, columns=self.X.columns,
                         index=range(100, 100+len(self.X.index)))
        metadata = pd.concat((self.X, x))

        exp_metadata = metadata.copy()
        ols('x1 + x2', self.Y, self.X)
        self.assertEqual(str(table), str(self.Y))
        self.assertEqual(str(metadata), str(exp_metadata))

    def test_ols_missing_metadata(self):
        # test to see if values in table get filtered out.
        # and that the original table doesn't change
        table = self.Y
        y = pd.DataFrame(self.Y.values, columns=self.Y.columns,
                         index=range(100, 100+len(self.Y.index)))

        table = pd.concat((self.Y, y))
        ids = np.arange(100, 100+len(self.X.index))
        x = pd.DataFrame([[np.nan] * len(self.X.columns)] * len(ids),
                         columns=self.X.columns, index=ids)

        metadata = pd.concat((self.X, x))
        model = ols('x1 + x2', table, metadata)
        model.fit()

        # test prediction
        exp = pd.DataFrame({'y1': self.r1_.predict(),
                            'y2': self.r2_.predict()},
                           index=self.Y.index)
        res = model.predict()

        pdt.assert_frame_equal(res, exp)

    def test_ols_test(self):

        model = ols('x1 + x2', self.Y, self.X)
        model.fit()

        # test pvalues
        exp = pd.DataFrame({'y1': self.r1_.pvalues,
                            'y2': self.r2_.pvalues})
        pdt.assert_frame_equal(model.pvalues, exp)

        # test coefficients
        exp = pd.DataFrame({'y1': self.r1_.params,
                            'y2': self.r2_.params})
        res = model.coefficients()
        pdt.assert_frame_equal(res, exp)

        # test residuals
        exp = pd.DataFrame({'y1': self.r1_.resid,
                            'y2': self.r2_.resid},
                           index=self.Y.index)
        res = model.residuals()
        pdt.assert_frame_equal(res, exp)

        # test prediction
        exp = pd.DataFrame({'y1': self.r1_.predict(),
                            'y2': self.r2_.predict()},
                           index=self.Y.index)
        res = model.predict()
        pdt.assert_frame_equal(res, exp)

        # make a small prediction
        fx = pd.DataFrame(
            [[1, 1, 1],
             [1, 1, 2]],
            columns=['Intercept', 'x1', 'x2'],
            index=['f1', 'f2'])

        rp1 = self.r1_.predict([[1, 1, 1],
                                [1, 1, 2]])
        rp2 = self.r2_.predict([[1, 1, 1],
                                [1, 1, 2]])
        exp = pd.DataFrame({'y1': rp1,
                            'y2': rp2},
                           index=['f1', 'f2'])

        res = model.predict(X=fx)
        pdt.assert_frame_equal(res,  exp)

        # test r2
        self.assertAlmostEqual(model.r2, 0.21981627865598752)

    def test_ols_ilr_inv_test(self):

        model = ols('x1 + x2', self.Y, self.X)
        model.fit()
        basis, _ = balance_basis(self.tree)
        # test pvalues
        exp = pd.DataFrame({'y1': self.r1_.pvalues,
                            'y2': self.r2_.pvalues})
        pdt.assert_frame_equal(model.pvalues, exp)

        # test coefficients
        exp = pd.DataFrame({'y1': self.r1_.params,
                            'y2': self.r2_.params})

        exp = pd.DataFrame(ilr_inv(exp, basis),
                           columns=['c', 'b', 'a'],
                           index=self.X.columns)

        res = model.coefficients(tree=self.tree)
        pdt.assert_frame_equal(res, exp)

        # test residuals
        exp = pd.DataFrame({'y1': self.r1_.resid,
                            'y2': self.r2_.resid},
                           index=self.Y.index)
        exp = pd.DataFrame(ilr_inv(exp, basis),
                           index=self.Y.index,
                           columns=['c', 'b', 'a'])
        res = model.residuals(tree=self.tree)
        pdt.assert_frame_equal(res, exp)

        # test prediction
        exp = pd.DataFrame({'y1': self.r1_.predict(),
                            'y2': self.r2_.predict()},
                           index=self.Y.index)
        exp = pd.DataFrame(ilr_inv(exp, basis),
                           index=self.Y.index,
                           columns=['c', 'b', 'a'])
        res = model.predict(tree=self.tree)
        pdt.assert_frame_equal(res, exp)

    def test_tvalues(self):
        model = ols('x1 + x2', self.Y, self.X)
        model.fit()

        exp = pd.DataFrame({'y1': self.r1_.tvalues,
                            'y2': self.r2_.tvalues})
        pdt.assert_frame_equal(model.tvalues, exp)

    def test_mse(self):
        model = ols('x1 + x2', self.Y, self.X)
        model.fit()

        exp = pd.Series({'y1': self.r1_.mse_resid,
                         'y2': self.r2_.mse_resid})
        pdt.assert_series_equal(model.mse, exp)

    def test_ess(self):
        model = ols('x1 + x2', self.Y, self.X)
        model.fit()

        exp = pd.Series({'y1': self.r1_.ess,
                         'y2': self.r2_.ess})
        pdt.assert_series_equal(model.ess, exp)

    def test_loo(self):
        model = ols('x1 + x2', self.Y, self.X)
        model.fit()
        res = model.loo()
        exp = pd.read_csv(get_data_path('loo.csv'), index_col=0)
        pdt.assert_frame_equal(res, exp)

    def test_kfold(self):
        model = ols('x1 + x2', self.Y, self.X)
        model.fit()
        res = model.kfold(9)
        exp = pd.read_csv(get_data_path('kfold.csv'), index_col=0)
        pdt.assert_frame_equal(res, exp)

    def test_lovo(self):
        model = ols('x1 + x2', self.Y, self.X)
        model.fit()
        res = model.lovo()
        exp = pd.read_csv(get_data_path('lovo.csv'), index_col=0)
        pdt.assert_frame_equal(res, exp)


class TestOLSCV(unittest.TestCase):
    """ Tests OLS regression with refactored matrix multiplication. """
    def setUp(self):
        np.random.seed(0)
        b01, b11, b21 = 1, 2, -3
        b02, b12, b22 = 2, -1, 4
        n = 50
        x1 = np.linspace(0, 10, n)
        x2 = np.linspace(10, 15, n)**2
        e = np.random.normal(size=n) * 10
        y1 = b01 + b11 * x1 + b21 * x2 + e
        e = np.random.normal(size=n) * 10
        y2 = b02 + b12 * x1 + b22 * x2 + e
        Y = pd.DataFrame(np.vstack((y1, y2)).T,
                         columns=['y1', 'y2'])

        B = pd.DataFrame([[b01, b11, b21],
                          [b02, b12, b22]])

        X = pd.DataFrame(
            np.vstack((np.ones(n), x1, x2)).T,
            columns=['Intercept', 'x1', 'x2'])

        self.Y = Y
        self.B = B
        self.X = X
        self.r1_ = smf.OLS(endog=y1, exog=X).fit()
        self.r2_ = smf.OLS(endog=y2, exog=X).fit()
        self.tree = TreeNode.read(['(c, (b,a)y2)y1;'])

    def test_loo(self):
        model = ols('x1 + x2', self.Y, self.X)
        model.fit()
        res = model.loo()
        exp = pd.read_csv(get_data_path('loo2.csv'), index_col=0)
        pdt.assert_frame_equal(res, exp)

    def test_kfold(self):
        model = ols('x1 + x2', self.Y, self.X)
        model.fit()
        res = model.kfold(9)
        exp = pd.read_csv(get_data_path('kfold2.csv'), index_col=0)
        pdt.assert_frame_equal(res, exp)

    def test_lovo(self):
        model = ols('x1 + x2', self.Y, self.X)
        model.fit()
        res = model.lovo()
        exp = pd.read_csv(get_data_path('lovo2.csv'), index_col=0)
        pdt.assert_frame_equal(res, exp)


if __name__ == "__main__":
    unittest.main()
