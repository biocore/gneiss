# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------
import unittest
from gneiss.plot import balance_boxplot, balance_barplots, proportion_plot
import numpy as np
import pandas as pd
import numpy.testing as npt
import matplotlib.pyplot as plt
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


class TestProportionPlot(unittest.TestCase):
    def setUp(self):
        self.table = pd.DataFrame({
            'A': [1, 1.2, 1.1, 2.1, 2.2, 2],
            'B': [9.9, 10, 10.1, 2, 2.4, 2.1],
            'C': [5, 3, 1, 2, 2, 3],
            'D': [5, 5, 5, 5, 5, 5],
        }, index=['S1', 'S2', 'S3', 'S4', 'S5', 'S6'])

        self.feature_metadata = pd.DataFrame({
            'A': ['k__foo', 'p__bar', 'c__', 'o__', 'f__', 'g__', 's__'],
            'B': ['k__foo', 'p__bar', 'c__', 'o__', 'f__', 'g__', 's__'],
            'C': ['k__poo', 'p__tar', 'c__', 'o__', 'f__', 'g__', 's__'],
            'D': ['k__poo', 'p__far', 'c__', 'o__', 'f__', 'g__', 's__']
        }, index=['kingdom', 'phylum', 'class', 'order',
                  'family', 'genus', 'species']).T

        self.metadata = pd.DataFrame({
            'groups': ['X', 'X', 'X', 'Y', 'Y', 'Y'],
            'dry': [1, 2, 3, 4, 5, 6]
        }, index=['S1', 'S2', 'S3', 'S4', 'S5', 'S6'])

    def test_proportion_plot(self):

        num_features = ['A', 'B']
        denom_features = ['C', 'D']
        ax1, ax2 = proportion_plot(self.table, self.metadata,
                                   num_features, denom_features,
                                   self.feature_metadata, 'groups', 'X', 'Y',
                                   taxa_level='phylum')
        res = np.vstack([l.get_xydata() for l in ax1.get_lines()])
        exp = np.array([[0.1863354, 0.],
                        [0.20529801, 0.],
                        [0.19254658, 1.],
                        [0.21794872, 1.],
                        [0.19230769, 2.],
                        [0.2484472, 2.],
                        [0.37267081, 3.],
                        [0.39735099, 3.]])
        npt.assert_allclose(res, exp)

        res = np.vstack([l.get_xydata() for l in ax2.get_lines()])
        exp = np.array([[0.08032129, 0.],
                        [0.0990566, 0.],
                        [0.437751, 1.],
                        [0.52358491, 1.],
                        [0.09433962, 2.],
                        [0.24096386, 2.],
                        [0.24096386, 3.],
                        [0.28301887, 3.]])
        npt.assert_allclose(res, exp)

        res = [l._text for l in ax2.get_yticklabels()]
        exp = ['p__bar', 'p__bar', 'p__tar', 'p__far']
        self.assertListEqual(res, exp)

    def test_proportion_plot_order(self):
        # tests for different ordering
        num_features = ['A', 'B']
        denom_features = ['D', 'C']
        ax1, ax2 = proportion_plot(self.table, self.metadata,
                                   num_features, denom_features,
                                   self.feature_metadata, 'groups', 'X', 'Y',
                                   taxa_level='phylum')
        res = np.vstack([l.get_xydata() for l in ax1.get_lines()])
        exp = np.array([[0.1863354, 0.],
                        [0.20529801, 0.],
                        [0.19254658, 1.],
                        [0.21794872, 1.],
                        [0.37267081, 2.],
                        [0.39735099, 2.],
                        [0.19230769, 3.],
                        [0.2484472, 3.]])
        npt.assert_allclose(res, exp, atol=1e-5)

        res = np.vstack([l.get_xydata() for l in ax2.get_lines()])
        exp = np.array([[0.08032129, 0.],
                        [0.0990566, 0.],
                        [0.437751, 1.],
                        [0.52358491, 1.],
                        [0.24096386, 2.],
                        [0.28301887, 2.],
                        [0.09433962, 3.],
                        [0.24096386, 3.]])
        npt.assert_allclose(res, exp, atol=1e-5)

        res = [l._text for l in ax2.get_yticklabels()]
        exp = ['p__bar', 'p__bar', 'p__far', 'p__tar']
        self.assertListEqual(res, exp)

    def test_proportion_plot_order_figure(self):
        # tests for different ordering
        fig, axes = plt.subplots(1, 2)

        num_features = ['A', 'B']
        denom_features = ['D', 'C']
        ax1, ax2 = proportion_plot(self.table, self.metadata,
                                   num_features, denom_features,
                                   self.feature_metadata, 'groups', 'X', 'Y',
                                   taxa_level='phylum', axes=axes)
        res = np.vstack([l.get_xydata() for l in ax1.get_lines()])
        exp = np.array([[0.1863354, 0.],
                        [0.20529801, 0.],
                        [0.19254658, 1.],
                        [0.21794872, 1.],
                        [0.37267081, 2.],
                        [0.39735099, 2.],
                        [0.19230769, 3.],
                        [0.2484472, 3.]])
        npt.assert_allclose(res, exp, atol=1e-2)

        res = np.vstack([l.get_xydata() for l in ax2.get_lines()])
        exp = np.array([[0.08032129, 0.],
                        [0.0990566, 0.],
                        [0.437751, 1.],
                        [0.52358491, 1.],
                        [0.24096386, 2.],
                        [0.28301887, 2.],
                        [0.09433962, 3.],
                        [0.24096386, 3.]])
        npt.assert_allclose(res, exp, atol=1e-2)

        res = [l._text for l in ax2.get_yticklabels()]
        exp = ['p__bar', 'p__bar', 'p__far', 'p__tar']
        self.assertListEqual(res, exp)


if __name__ == '__main__':
    unittest.main()
