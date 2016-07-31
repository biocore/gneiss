#!/usr/bin/env python

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
import statsmodels.formula.api as smf
import unittest
from gneiss._summary import RegressionResults
from skbio.stats.composition import _gram_schmidt_basis, ilr_inv
from skbio import TreeNode
from gneiss._formula import ols


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


if __name__ == '__main__':
    unittest.main()
