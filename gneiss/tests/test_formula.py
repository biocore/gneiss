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
import unittest
from skbio.stats.composition import ilr_inv
from skbio import TreeNode
from gneiss._formula import ols, mixedlm


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

    def test_mixedlm(self):
        res = mixedlm('time', self.table, self.metadata, self.tree, groups=patient)

        pass

if __name__ == '__main__':
    unittest.main()
