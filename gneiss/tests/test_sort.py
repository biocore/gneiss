# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------
import numpy as np
import pandas as pd
import unittest
from gneiss.sort import (niche_sort, mean_niche_estimator,
                         ladderize, gradient_sort)
import pandas.util.testing as pdt
from skbio import TreeNode


class TestSort(unittest.TestCase):
    def setUp(self):
        pass

    def test_mean_niche_estimator1(self):
        gradient = pd.Series(
            [1, 2, 3, 4, 5],
            index=['s1', 's2', 's3', 's4', 's5'])
        values = pd.Series(
            [1, 1, 0, 0, 0],
            index=['s1', 's2', 's3', 's4', 's5'])
        m = mean_niche_estimator(values, gradient)
        self.assertEqual(m, 1.5)

    def test_mean_niche_estimator2(self):
        gradient = pd.Series(
            [1, 2, 3, 4, 5],
            index=['s1', 's2', 's3', 's4', 's5'])
        values = pd.Series(
            [1, 3, 0, 0, 0],
            index=['s1', 's2', 's3', 's4', 's5'])
        m = mean_niche_estimator(values, gradient)
        self.assertEqual(m, 1.75)

    def test_mean_niche_estimator_frame(self):
        gradient = pd.Series(
            [1, 2, 3, 4, 5],
            index=['s1', 's2', 's3', 's4', 's5'])
        values = pd.DataFrame(
            np.array([[1, 3, 0, 0, 0],
                      [1, 3, 0, 0, 0]]).T,
            index=['s1', 's2', 's3', 's4', 's5'],
            columns=['o1', 'o2'])
        m = mean_niche_estimator(values, gradient)
        exp = pd.Series([1.75, 1.75], index=['o1', 'o2'])
        pdt.assert_series_equal(m, exp)

    def test_mean_niche_estimator_bad_length(self):
        gradient = pd.Series(
            [1, 2, 3, 4, 5],
            index=['s1', 's2', 's3', 's4', 's5'])
        values = pd.Series(
            [1, 3, 0, 0, 0, 0],
            index=['s1', 's2', 's3', 's4', 's5', 's6'])

        with self.assertRaises(ValueError):
            mean_niche_estimator(values, gradient)

    def test_mean_niche_estimator_missing(self):
        gradient = pd.Series(
            [1, 2, 3, 4, np.nan],
            index=['s1', 's2', 's3', 's4', 's5'])
        values = pd.Series(
            [1, 3, 0, 0, 0],
            index=['s1', 's2', 's3', 's4', 's5'])

        with self.assertRaises(ValueError):
            mean_niche_estimator(values, gradient)

    def test_basic_niche_sort(self):
        table = pd.DataFrame(
            [[1, 1, 0, 0, 0],
             [0, 1, 1, 0, 0],
             [0, 0, 1, 1, 0],
             [0, 0, 0, 1, 1]],
            columns=['s1', 's2', 's3', 's4', 's5'],
            index=['o1', 'o2', 'o3', 'o4']).T
        gradient = pd.Series(
            [1, 2, 3, 4, 5],
            index=['s1', 's2', 's3', 's4', 's5'])
        res_table = niche_sort(table, gradient)
        pdt.assert_frame_equal(table, res_table)

    def test_basic_niche_sort_error(self):
        table = pd.DataFrame(
            [[1, 1, 0, 0, 0],
             [0, 1, 1, 0, 0],
             [0, 0, 1, 1, 0],
             [0, 0, 0, 1, 1]],
            columns=['s1', 's2', 's3', 's4', 's5'],
            index=['o1', 'o2', 'o3', 'o4']).T
        gradient = pd.Series(
            [1, 2, 3, 4, 5],
            index=['s1', 's2', 's3', 's4', 's5'])
        with self.assertRaises(ValueError):
            niche_sort(table, gradient, niche_estimator='rawr')

    def test_basic_niche_sort_scrambled(self):
        # Swap samples s1 and s2 and features o1 and o2 to see if this can
        # obtain the original table structure.
        table = pd.DataFrame(
            [[1, 0, 1, 0, 0],
             [1, 1, 0, 0, 0],
             [0, 0, 1, 1, 0],
             [0, 0, 0, 1, 1]],
            columns=['s2', 's1', 's3', 's4', 's5'],
            index=['o2', 'o1', 'o3', 'o4']).T

        gradient = pd.Series(
            [2, 1, 3, 4, 5],
            index=['s2', 's1', 's3', 's4', 's5'])

        exp_table = pd.DataFrame(
            [[1, 1, 0, 0, 0],
             [0, 1, 1, 0, 0],
             [0, 0, 1, 1, 0],
             [0, 0, 0, 1, 1]],
            columns=['s1', 's2', 's3', 's4', 's5'],
            index=['o1', 'o2', 'o3', 'o4']).T

        res_table = niche_sort(table, gradient)

        pdt.assert_frame_equal(exp_table, res_table)

    def test_basic_niche_sort_lambda(self):
        table = pd.DataFrame(
            [[1, 1, 0, 0, 0],
             [0, 0, 1, 1, 0],
             [0, 1, 1, 0, 0],
             [0, 0, 0, 1, 1]],
            columns=['s1', 's2', 's3', 's4', 's5'],
            index=['o1', 'o3', 'o2', 'o4']).T
        gradient = pd.Series(
            [1, 2, 3, 4, 5],
            index=['s1', 's2', 's3', 's4',  's5'])

        exp_table = pd.DataFrame(
            [[1, 1, 0, 0, 0],
             [0, 1, 1, 0, 0],
             [0, 0, 1, 1, 0],
             [0, 0, 0, 1, 1]],
            columns=['s1', 's2', 's3', 's4', 's5'],
            index=['o1', 'o2', 'o3', 'o4']).T

        def _dumb_estimator(v, gradient):
            v[v > 0] = 1
            values = v / v.sum()
            return np.dot(gradient, values)

        res_table = niche_sort(table, gradient,
                               niche_estimator=_dumb_estimator)
        pdt.assert_frame_equal(exp_table, res_table)

    def test_basic_niche_sort_immutable(self):
        # Swap samples s1 and s2 and features o1 and o2 to see if this can
        # obtain the original table structure.
        table = pd.DataFrame(
            [[1, 0, 1, 0, 0],
             [1, 1, 0, 0, 0],
             [0, 0, 1, 1, 0],
             [0, 0, 0, 1, 1]],
            columns=['s2', 's1', 's3', 's4', 's5'],
            index=['o2', 'o1', 'o3', 'o4']).T

        gradient = pd.Series(
            [2, 1, 3, 4, 5],
            index=['s2', 's1', 's3', 's4', 's5'])

        exp_table = pd.DataFrame(
            [[1, 0, 1, 0, 0],
             [1, 1, 0, 0, 0],
             [0, 0, 1, 1, 0],
             [0, 0, 0, 1, 1]],
            columns=['s2', 's1', 's3', 's4', 's5'],
            index=['o2', 'o1', 'o3', 'o4']).T

        exp_gradient = pd.Series(
            [2, 1, 3, 4, 5],
            index=['s2', 's1', 's3', 's4', 's5'])

        niche_sort(table, gradient)
        pdt.assert_frame_equal(exp_table, table)
        pdt.assert_series_equal(exp_gradient, gradient)

    def test_ladderize1(self):
        # Makes sure that 1 subtree is ordered
        tree = TreeNode.read([u'((a,b)c, ((g,h)e,f)d)r;'])
        exp = '((a,b)c,(f,(g,h)e)d)r;\n'
        res = str(ladderize(tree))
        self.assertEqual(exp, res)

    def test_ladderize2(self):
        # Makes sure that 2 subtrees are ordered
        tree = TreeNode.read([u'(((n,m)a,b)c, ((g,(i,j)h)e,f)d)r;'])
        exp = '((b,(n,m)a)c,(f,(g,(i,j)h)e)d)r;\n'
        res = str(ladderize(tree))
        self.assertEqual(exp, res)

    def test_ladderize_descending(self):
        # Makes sure that 2 subtrees are ordered
        tree = TreeNode.read([u'(((n,m)a,b)c, ((g,(i,j)h)e,f)d)r;'])
        exp = '((((j,i)h,g)e,f)d,((m,n)a,b)c)r;\n'
        res = str(ladderize(tree, ascending=False))
        self.assertEqual(exp, res)

    def test_gradient_sort(self):
        # Makes sure that the tree is sorted according
        # a pre-set ordering
        tree = TreeNode.read([u'((a,b)c, ((g,h)e,f)d)r;'])
        exp = '(((g,h)e,f)d,(a,b)c)r;\n'
        x = pd.Series({'f': 3, 'g': 1, 'h': 2, 'a': 4, 'b': 5})
        res = str(gradient_sort(tree, x))
        self.assertEqual(exp, res)

    def test_gradient_sort_descending(self):
        # Makes sure that the tree is sorted according
        # a pre-set ordering in descending order
        tree = TreeNode.read([u'((a,b)c, ((g,h)e,f)d)r;'])
        exp = '((b,a)c,(f,(h,g)e)d)r;\n'
        x = pd.Series({'f': 3, 'g': 1, 'h': 2, 'a': 4, 'b': 5})
        res = str(gradient_sort(tree, x, ascending=False))
        self.assertEqual(exp, res)

    def test_gradient_sort_error(self):
        # Makes sure that the tree is sorted according
        # a pre-set ordering
        tree = TreeNode.read([u'((a,b)c, ((g,h)e,f)d)r;'])
        x = pd.Series({'f': 'x', 'g': 'y', 'h': 'z', 'a': 'u', 'b': 'dz'})
        with self.assertRaises(ValueError):
            gradient_sort(tree, x)


if __name__ == '__main__':
    unittest.main()
