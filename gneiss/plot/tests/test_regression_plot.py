# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------
import unittest
import os
import shutil

import numpy as np
import pandas as pd
import numpy.testing as npt

from skbio import TreeNode
from skbio.util import get_data_path

from gneiss.plot._regression_plot import ols_summary, lme_summary
from gneiss.regression import ols, mixedlm


class TestOLS_Summary(unittest.TestCase):

    def setUp(self):
        A = np.array  # aliasing for the sake of pep8
        self.table = pd.DataFrame({
            's1': A([1., 1.]),
            's2': A([1., 2.]),
            's3': A([1., 3.]),
            's4': A([1., 4.]),
            's5': A([1., 5.])},
            index=['Y2', 'Y1']).T
        self.tree = TreeNode.read(['(c, (b,a)Y2)Y1;'])
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
        sy = np.vstack((-y/10, -y)).T
        self.y = pd.DataFrame(sy, columns=['y0', 'y1'])
        self.t2 = TreeNode.read([r"((a,b)y1,c)y0;"])

        self.results = "results"
        os.mkdir(self.results)

    def tearDown(self):
        shutil.rmtree(self.results)

    def test_visualization(self):
        res = ols(formula="x1 + x2 + x3 + x4",
                  table=self.y, metadata=self.x)
        res.fit()

        ols_summary(self.results, res, tree=self.t2)
        fp = os.path.join(self.results, 'pvalues.csv')
        self.assertTrue(os.path.exists(fp))
        fp = os.path.join(self.results, 'coefficients.csv')
        self.assertTrue(os.path.exists(fp))
        fp = os.path.join(self.results, 'predicted.csv')
        self.assertTrue(os.path.exists(fp))
        fp = os.path.join(self.results, 'residuals.csv')
        self.assertTrue(os.path.exists(fp))

        index_fp = os.path.join(self.results, 'index.html')
        self.assertTrue(os.path.exists(index_fp))

        with open(index_fp, 'r') as fh:
            html = fh.read()
            self.assertIn('<h1>Simplicial Linear Regression Summary</h1>',
                          html)
            self.assertIn('<th>Coefficients</th>\n', html)
            self.assertIn('<th>Predicted Balances</th>\n', html)
            self.assertIn('<th>Residuals</th>\n', html)


class TestLME_Summary(unittest.TestCase):

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
        df["Y1"] = endog + 2 * 2
        df["Y2"] = endog
        df["groups"] = groups
        df["x1"] = exog[:, 0]
        df["x2"] = exog[:, 1]

        self.tree = TreeNode.read(['(c, (b,a)Y2)Y1;'])
        self.table = df[["Y1", "Y2"]]
        self.metadata = df[['x1', 'x2', 'groups']]

        self.results = "results"
        if not os.path.exists(self.results):
            os.mkdir(self.results)

    def tearDown(self):
        shutil.rmtree(self.results)

    def test_visualization(self):
        model = mixedlm("x1 + x2", self.table, self.metadata,
                        groups="groups")
        model.fit()
        lme_summary(self.results, model, self.tree)
        pvals = pd.read_csv(os.path.join(self.results, 'pvalues.csv'),
                            index_col=0)
        coefs = pd.read_csv(os.path.join(self.results, 'coefficients.csv'),
                            index_col=0)
        pred = pd.read_csv(os.path.join(self.results, 'predicted.csv'),
                           index_col=0)
        resid = pd.read_csv(os.path.join(self.results, 'residuals.csv'),
                            index_col=0)

        exp_pvals = pd.DataFrame({
            'Intercept': {'Y1': 4.8268860492262526e-236,
                          'Y2': 0.099411090631406948},
            'Group Var': {'Y1': 4.4193804668281966e-05,
                          'Y2': 4.4193804668280984e-05},
            'x1': {'Y1': 3.9704936434633392e-35,
                   'Y2': 3.9704936434628853e-35},
            'x2': {'Y1': 3.56912071867573e-30,
                   'Y2': 3.56912071867573e-30}}).sort_index(axis=1)
        pvals = pvals.sort_index(axis=0).sort_index(axis=1)
        exp_pvals = exp_pvals.sort_index(axis=0).sort_index(axis=1)

        npt.assert_allclose(pvals, exp_pvals, rtol=1e-5)

        exp_coefs = pd.DataFrame({
            'Intercept': {'Y1': 4.2115280233151946,
                          'Y2': 0.211528023315187},
            'Group Var': {'Y1': 0.093578639287859755,
                          'Y2': 0.093578639287860019},
            'x1': {'Y1': 1.0220072967452645,
                   'Y2': 1.0220072967452651},
            'x2': {'Y1': 0.92487193877761575,
                   'Y2': 0.92487193877761564}}
        ).sort_index(axis=1)

        npt.assert_allclose(coefs.sort_index(axis=0),
                            exp_coefs.sort_index(axis=0),
                            rtol=1e-2, atol=1e-2)

        exp_resid = pd.read_csv(get_data_path('exp_resid.csv'), index_col=0)
        npt.assert_allclose(resid, exp_resid.T, rtol=1e-2, atol=1e-2)

        exp_pred = pd.read_csv(get_data_path('exp_pred.csv'), index_col=0)
        npt.assert_allclose(pred, exp_pred.T, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    unittest.main()
