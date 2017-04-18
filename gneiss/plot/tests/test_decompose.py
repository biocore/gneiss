# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------
import unittest
from gneiss.plot._decompose import balance_boxplot, balance_barplots
import numpy as np
import pandas as pd
import numpy.testing as npt
from skbio import TreeNode


class TestBoxplot(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'y': [-2, -2.2, -1.8, -1.5, -1, 1, 1.5, 2, 2.2, 1.8],
            'group': ['a', 'a', 'a', 'a', 'a', 'b', 'b', 'b', 'b', 'b']},
        )
        self.tree = TreeNode.read(['((c, d)z, (b,a)x)y;'])
        self.feature_df = pd.DataFrame(
            {
                'type': ['tomato', 'carrots', 'apple', 'bacon'],
                'food': ['vegatable', 'vegatable', 'fruit', 'meat'],
                'seed': ['yes', 'no', 'yes', 'no']
            },
            index=["a", "b", "c", "d"]
        )

    def test_basic_boxplot(self):
        a = balance_boxplot('y', x='group', data=self.df)
        res = np.vstack([i._xy for i in a.get_lines()])
        exp = np.array([[0., -2.],
                        [0., -2.2],
                        [0., -1.5],
                        [0., -1.],
                        [-0.2, -2.2],
                        [0.2, -2.2],
                        [-0.2, -1.],
                        [0.2, -1.],
                        [-0.4, -1.8],
                        [0.4, -1.8],
                        [1., 1.5],
                        [1., 1.],
                        [1., 2.],
                        [1., 2.2],
                        [0.8, 1.],
                        [1.2, 1.],
                        [0.8, 2.2],
                        [1.2, 2.2],
                        [0.6, 1.8],
                        [1.4, 1.8]])
        npt.assert_allclose(res, exp)

    def test_basic_barplot(self):
        ax_denom, ax_num = balance_barplots(self.tree, 'y', header='food',
                                            feature_metadata=self.feature_df)
        res_num = np.vstack([i._xy for i in ax_num.get_lines()])
        res_denom = np.vstack([i._xy for i in ax_denom.get_lines()])
        exp_num = np.array([[np.nan, 0.],
                            [np.nan, 0.]])
        exp_denom = np.array([[np.nan, 0.],
                              [np.nan, 0.],
                              [np.nan, 1.],
                              [np.nan, 1.]])
        npt.assert_allclose(res_num, exp_num)
        npt.assert_allclose(res_denom, exp_denom)


if __name__ == '__main__':
    unittest.main()
