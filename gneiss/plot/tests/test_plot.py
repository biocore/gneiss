# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the GPLv3 License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------
import unittest
import os
import shutil

import numpy as np
import pandas as pd

import numpy as np
import numpy.testing as npt

from skbio import TreeNode
from skbio.stats.composition import ilr_inv
from skbio.util import get_data_path

from gneiss.plot._plot import ols_summary, lme_summary
from gneiss.regression import ols, mixedlm


class TestOLS_Summary(unittest.TestCase):

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

        self.results = "results"
        os.mkdir(self.results)

    def tearDown(self):
        shutil.rmtree(self.results)

    def test_visualization(self):
        res = ols(formula="x1 + x2 + x3 + x4",
                  table=self.y, metadata=self.x, tree=self.t2)
        res.fit()

        ols_summary(self.results, res)
        pvals = pd.read_csv(os.path.join(self.results, 'pvalues.csv'),
                            index_col=0)
        coefs = pd.read_csv(os.path.join(self.results, 'coefficients.csv'),
                            index_col=0)
        pred = pd.read_csv(os.path.join(self.results, 'predicted.csv'),
                           index_col=0)
        resid = pd.read_csv(os.path.join(self.results, 'residuals.csv'),
                            index_col=0)

        exp_pred = pd.DataFrame({
            'y0': {0: -0.53375121547306381,
                   1: -0.56479853016207482,
                   2: -0.56287346890240741,
                   3: -0.54189204731941831,
                   4: -0.51324876614124992,
                   5: -0.48580516711594918,
                   6: -0.46588315729838481,
                   7: -0.45726500901030648,
                   8: -0.46118573050287187,
                   9: -0.47632066813918106,
                   10: -0.49878455279984207,
                   11: -0.52212577764307233,
                   12: -0.53732163007547018,
                   13: -0.53276780094653364,
                   14: -0.49427170901103434},
            'y1': {0: -5.3374587490426801,
                   1: -5.6479395254526388,
                   2: -5.628692727739514,
                   3: -5.4188785121568728,
                   4: -5.1324342565916066,
                   5: -4.8580135254968413,
                   6: -4.6587877057054454,
                   7: -4.5725985939212412,
                   8: -4.6118058093989722,
                   9: -4.7631647231182699,
                   10: -4.9877959410043564,
                   11: -5.2212158195661642,
                   12: -5.3731686226401827,
                   13: -5.3276265175104554,
                   14: -4.942667506421965}})
        npt.assert_allclose(exp_pred.values, pred.values, rtol=1e-5)

        exp_coefs = pd.DataFrame({
            'Intercept': {'y0': 6880999561.7020159,
                          'y1': 68809995617.020004},
            'x1': {'y0': 676465286.62179089,
                   'y1': 6764652866.2178936},
            'x2': {'y0': 610204064.32702351,
                   'y1': 6102040643.2702208},
            'x3': {'y0': -7497970910.8040514,
                   'y1': -74979709108.040298},
            'x4': {'y0': 26313750.43187603,
                   'y1': 263137504.31875956}})
        npt.assert_allclose(exp_coefs.values, coefs.values, rtol=1e-5)

        exp_resid = pd.DataFrame({
            'y0': {0: -0.05693401912370244,
                   1: 0.10695167979147802,
                   2: 0.043549965263444679,
                   3: -0.10733300196780859,
                   4: -0.10239623711106705,
                   5: 0.15082282297327071,
                   6: -0.065724199795309968,
                   7: 0.031846373671398198,
                   8: 0.026929958766511719,
                   9: -0.013407601943539682,
                   10: 0.031553548285065736,
                   11: -0.080353914812739569,
                   12: -0.00012764772886153519,
                   13: 0.054894158986049046,
                   14: -0.02024886341379073},
            'y1': {0: -0.56939359692498392,
                   1: 1.0694710217466721,
                   2: 0.4354576913498871,
                   3: -1.0733719807153905,
                   4: -1.0240157759315673,
                   5: 1.5081900840700544,
                   6: -0.65728586523150234,
                   7: 0.318412240532159,
                   8: 0.26924809203537148,
                   9: -0.13411797770893941,
                   10: 0.31548589585659403,
                   11: -0.80358110499195856,
                   12: -0.0013241554031324654,
                   13: 0.54889009790560728,
                   14: -0.20253821782628822}})
        npt.assert_allclose(exp_resid.values, resid.values, rtol=1e-5)

        exp_pvals = pd.DataFrame({
            'Intercept': {'y0': 0.3193097383026624,
                          'y1': 0.31931029350376261},
            'x1': {'y0': 0.31931130074025166,
                   'y1': 0.31931185594151867},
            'x2': {'y0': 0.31929793802591028,
                   'y1': 0.3192984932257481},
            'x3': {'y0': 0.31930876472902192,
                   'y1': 0.31930931993001832},
            'x4': {'y0': 0.31931786743864193,
                   'y1': 0.31931842264061172}})

        npt.assert_allclose(exp_pvals.values, pvals.values, rtol=1e-5)
        index_fp = os.path.join(self.results, 'index.html')
        self.assertTrue(os.path.exists(index_fp))

        with open(index_fp, 'r') as fh:
            html = fh.read()
            self.assertIn('<h1>Simplicial Linear Regression Summary</h1>',
                          html)
            self.assertIn('<th>Relative importance</th>', html)
            self.assertIn('<th>Cross Validation</th>', html)
            self.assertIn('<th>Coefficients</th>\n', html)
            self.assertIn('<th>Raw Balances</th>\n', html)
            self.assertIn('<th>Predicted Proportions</th>\n', html)
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
        df["y1"] = endog
        df["y2"] = endog + 2 * 2
        df["groups"] = groups
        df["x1"] = exog[:, 0]
        df["x2"] = exog[:, 1]

        self.tree = TreeNode.read(['(c, (b,a)Y2)Y1;'])
        iv = ilr_inv(df[["y1", "y2"]].values)
        self.table = pd.DataFrame(iv, columns=['a', 'b', 'c'])
        self.metadata = df[['x1', 'x2', 'groups']]

        self.results = "results"
        os.mkdir(self.results)

    def tearDown(self):
        shutil.rmtree(self.results)

    def test_visualization(self):
        model = mixedlm("x1 + x2", self.table, self.metadata, self.tree,
                      groups="groups")
        model.fit()
        lme_summary(self.results, model)
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
            'groups RE': {'Y1': 4.4193804668281966e-05,
                          'Y2': 4.4193804668280984e-05},
            'x1': {'Y1': 3.9704936434633392e-35,
                   'Y2': 3.9704936434628853e-35},
            'x2': {'Y1': 3.56912071867573e-30,
                   'Y2': 3.56912071867573e-30}})
        npt.assert_allclose(pvals, exp_pvals, rtol=1e-5)

        exp_coefs = pd.DataFrame({
            'Intercept': {'Y1': 4.2115280233151946,
                          'Y2': 0.211528023315187},
            'groups RE': {'Y1': 0.093578639287859755,
                          'Y2': 0.093578639287860019},
            'x1': {'Y1': 1.0220072967452645,
                   'Y2': 1.0220072967452651},
            'x2': {'Y1': 0.92487193877761575,
                   'Y2': 0.92487193877761564}}
        )

        exp_resid = pd.read_csv(get_data_path('exp_resid.csv'), index_col=0)
        npt.assert_allclose(resid, exp_resid, rtol=1e-5)

        exp_pred = pd.read_csv(get_data_path('exp_pred.csv'), index_col=0)
        npt.assert_allclose(pred, exp_pred, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
