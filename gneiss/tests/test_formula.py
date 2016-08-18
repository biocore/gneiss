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

    def test_ols_empty(self):
        pass

if __name__ == '__main__':
    unittest.main()
