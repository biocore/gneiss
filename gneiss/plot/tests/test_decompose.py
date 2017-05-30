# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------
import unittest
from gneiss.plot import balance_boxplot, balance_barplots
import numpy as np
import pandas as pd
import numpy.testing as npt
from skbio import TreeNode


class TestBoxplot(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'y': [-2, -2.2, -1.8, -1.5, -1, 1, 1.5, 2, 2.2, 1.8],
            'group': ['a', 'a', 'a', 'a', 'a', 'b', 'b', 'b', 'b', 'b'],
            'hue': ['0', '1', '0', '1', '0', '1', '0', '1', '0', '1']}
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
        a = balance_boxplot('y', y='group', data=self.df)
        res = np.vstack([i._xy for i in a.get_lines()])
        exp = np.array([[-2., 0.],
                        [-2.2, 0.],
                        [-1.5, 0.],
                        [-1., 0.],
                        [-2.2, -0.2],
                        [-2.2, 0.2],
                        [-1., -0.2],
                        [-1., 0.2],
                        [-1.8, -0.4],
                        [-1.8, 0.4],
                        [1.5, 1.],
                        [1., 1.],
                        [2., 1.],
                        [2.2, 1.],
                        [1., 0.8],
                        [1., 1.2],
                        [2.2, 0.8],
                        [2.2, 1.2],
                        [1.8, 0.6],
                        [1.8, 1.4]])
        npt.assert_allclose(res, exp)

    def test_basic_hue_boxplot(self):
        a = balance_boxplot('y', y='group', hue='hue', data=self.df)
        res = np.vstack([i._xy for i in a.get_lines()])
        exp = np.array([[-1.9, -0.2],
                        [-2., -0.2],
                        [-1.4, -0.2],
                        [-1., -0.2],
                        [-2., -0.298],
                        [-2., -0.102],
                        [-1., -0.298],
                        [-1., -0.102],
                        [-1.8, -0.396],
                        [-1.8, -0.004],
                        [-2.025, 0.2],
                        [-2.2, 0.2],
                        [-1.675, 0.2],
                        [-1.5, 0.2],
                        [-2.2, 0.102],
                        [-2.2, 0.298],
                        [-1.5, 0.102],
                        [-1.5, 0.298],
                        [-1.85, 0.004],
                        [-1.85, 0.396],
                        [1.675, 0.8],
                        [1.5, 0.8],
                        [2.025, 0.8],
                        [2.2, 0.8],
                        [1.5, 0.702],
                        [1.5, 0.898],
                        [2.2, 0.702],
                        [2.2, 0.898],
                        [1.85, 0.604],
                        [1.85, 0.996],
                        [1.4, 1.2],
                        [1., 1.2],
                        [1.9, 1.2],
                        [2., 1.2],
                        [1., 1.102],
                        [1., 1.298],
                        [2., 1.102],
                        [2., 1.298],
                        [1.8, 1.004],
                        [1.8, 1.396]])
        npt.assert_allclose(res, exp)

    def test_basic_barplot(self):
        ax_denom, ax_num = balance_barplots(self.tree, 'y', header='food',
                                            feature_metadata=self.feature_df)


if __name__ == '__main__':
    unittest.main()
