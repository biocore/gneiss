import numpy as np
import pandas as pd
import unittest
from gneiss.sort import nichesort, mean_niche_estimator
import pandas.util.testing as pdt


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

    def test_basic_nichesort(self):
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
        res_table = nichesort(table, gradient)
        pdt.assert_frame_equal(table, res_table)

    def test_basic_nichesort_scrambled(self):

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

        res_table = nichesort(table, gradient)

        pdt.assert_frame_equal(exp_table, res_table)

    def test_basic_nichesort_lambda(self):
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

        res_table = nichesort(table, gradient, niche_estimator=_dumb_estimator)
        pdt.assert_frame_equal(exp_table, res_table)


if __name__ == '__main__':
    unittest.main()
