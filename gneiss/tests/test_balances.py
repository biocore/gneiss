from __future__ import absolute_import, division, print_function
import unittest
import numpy as np
import pandas as pd
import numpy.testing as npt
from gneiss.balances import (balance_basis, _count_matrix,
                             _balance_basis, _attach_balances,
                             balanceplot)
from gneiss.layouts import default_layout
from skbio import TreeNode
from skbio.util import get_data_path


class TestPlot(unittest.TestCase):

    def test__attach_balances(self):
        tree = TreeNode.read([u"(a,b);"])
        balances = np.array([10])
        res_tree = _attach_balances(balances, tree)
        self.assertEqual(res_tree.weight, 10)

    def test__attach_balances_level_order(self):
        tree = TreeNode.read([u"((a,b)c,d)r;"])
        balances = np.array([10, -10])
        res_tree = _attach_balances(balances, tree)
        self.assertEqual(res_tree.weight, 10)
        self.assertEqual(res_tree.children[0].weight, -10)

    def test__attach_balances_bad_index(self):
        tree = TreeNode.read([u"((a,b)c,d)r;"])
        balances = np.array([10])
        with self.assertRaises(IndexError):
            _attach_balances(balances, tree)

    def test__attach_balances_series(self):
        tree = TreeNode.read([u"((a,b)c,d)r;"])
        balances = pd.Series([10, -10], index=['r', 'c'])
        res_tree = _attach_balances(balances, tree)
        self.assertEqual(res_tree.weight, 10)

    def test__attach_balances_series_bad(self):
        tree = TreeNode.read([u"((a,b)c,d)r;"])
        balances = pd.Series([10, -10])
        with self.assertRaises(KeyError):
            _attach_balances(balances, tree)

    def test_balanceplot(self):
        tree = TreeNode.read([u"((a,b)c,d)r;"])
        balances = np.array([10, -10])
        tr, ts = balanceplot(balances, tree)
        self.assertEqual(ts.mode, 'c')
        self.assertEqual(ts.layout_fn[0], default_layout)


class TestBalances(unittest.TestCase):

    def test_count_matrix_base_case(self):
        tree = u"(a,b);"
        t = TreeNode.read([tree])
        res, _, _ = _count_matrix(t)

        exp = np.array([1, 1, 0, 0])
        npt.assert_equal(res[0], exp)

        exp = np.array([0, 0, 0, 0])
        npt.assert_equal(res[1], exp)

        exp = np.array([0, 0, 0, 0])
        npt.assert_equal(res[2], exp)

    def test_count_matrix_unbalanced(self):
        tree = u"((a,b)c, d);"
        t = TreeNode.read([tree])
        res, _, _ = _count_matrix(t)

        exp = np.array([1, 2, 0, 0])
        npt.assert_equal(res[0], exp)
        exp = np.array([1, 1, 1, 0])
        npt.assert_equal(res[1], exp)

        exp = np.array([0, 0, 0, 0])
        npt.assert_equal(res[2], exp)
        npt.assert_equal(res[3], exp)
        npt.assert_equal(res[4], exp)

    def test_count_matrix_singleton_error(self):
        with self.assertRaises(ValueError):
            tree = u"(((a,b)c, d)root);"
            t = TreeNode.read([tree])
            _count_matrix(t)

    def test_count_matrix_trifurcating_error(self):
        with self.assertRaises(ValueError):
            tree = u"((a,b,e)c, d);"
            t = TreeNode.read([tree])
            _count_matrix(t)

    def test__balance_basis_base_case(self):
        tree = u"(a,b);"
        t = TreeNode.read([tree])

        exp_basis = np.array([[-np.sqrt(1. / 2), np.sqrt(1. / 2)]])
        exp_keys = [t]
        res_basis, res_keys = _balance_basis(t)

        npt.assert_allclose(exp_basis, res_basis)
        self.assertListEqual(exp_keys, res_keys)

    def test__balance_basis_unbalanced(self):
        tree = u"((a,b)c, d);"
        t = TreeNode.read([tree])

        exp_basis = np.array([[-np.sqrt(1. / 6), -np.sqrt(1. / 6),
                               np.sqrt(2. / 3)],
                              [-np.sqrt(1. / 2), np.sqrt(1. / 2), 0]
                              ])
        exp_keys = [t, t[0]]
        res_basis, res_keys = _balance_basis(t)

        npt.assert_allclose(exp_basis, res_basis)
        self.assertListEqual(exp_keys, res_keys)

    def test_balance_basis_base_case(self):
        tree = u"(a,b);"
        t = TreeNode.read([tree])
        exp_keys = [t]
        exp_basis = np.array([0.19557032, 0.80442968])
        res_basis, res_keys = balance_basis(t)

        npt.assert_allclose(exp_basis, res_basis)
        self.assertListEqual(exp_keys, res_keys)

    def test_balance_basis_unbalanced(self):
        tree = u"((a,b)c, d);"
        t = TreeNode.read([tree])
        exp_keys = [t, t[0]]
        exp_basis = np.array([[0.18507216, 0.18507216, 0.62985567],
                              [0.14002925, 0.57597535, 0.28399541]])

        res_basis, res_keys = balance_basis(t)

        npt.assert_allclose(exp_basis, res_basis)
        self.assertListEqual(exp_keys, list(res_keys))

    def test_balance_basis_large1(self):
        fname = get_data_path('large_tree.nwk',
                              subfolder='data')
        t = TreeNode.read(fname)
        # note that the basis is in reverse level order
        exp_basis = np.loadtxt(
            get_data_path('large_tree_basis.txt',
                          subfolder='data'))
        res_basis, res_keys = balance_basis(t)

        exp_basis = exp_basis[:, ::-1]
        print(exp_basis.shape)
        for i in range(len(res_basis)):
            print(i)
            npt.assert_allclose(exp_basis[i], res_basis[i])

        npt.assert_allclose(exp_basis[:, ::-1], res_basis)


if __name__ == "__main__":
    unittest.main()
