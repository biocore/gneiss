# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

from __future__ import absolute_import, division, print_function
import unittest
import numpy as np
import numpy.testing as npt
from gneiss.balances import (balance_basis, _count_matrix,
                             _balance_basis, sparse_balance_basis)
from skbio import TreeNode
from skbio.util import get_data_path
from scipy.sparse import coo_matrix


def assert_coo_allclose(res, exp, rtol=1e-7, atol=1e-7):
    res_data = np.vstack((res.row, res.col, res.data)).T
    exp_data = np.vstack((exp.row, exp.col, exp.data)).T

    # sort by row and col
    res_data = res_data[res_data[:, 1].argsort()]
    res_data = res_data[res_data[:, 0].argsort()]
    exp_data = exp_data[exp_data[:, 1].argsort()]
    exp_data = exp_data[exp_data[:, 0].argsort()]
    npt.assert_allclose(res_data, exp_data, rtol=rtol, atol=atol)


class TestSparseBalances(unittest.TestCase):

    def test_sparse_balance_basis_base_case(self):
        tree = u"(a,b);"
        t = TreeNode.read([tree])

        exp_basis = coo_matrix(
            np.array([[-np.sqrt(1. / 2),
                       np.sqrt(1. / 2)]]))
        exp_keys = [t.name]
        res_basis, res_keys = sparse_balance_basis(t)

        assert_coo_allclose(exp_basis, res_basis)
        self.assertListEqual(exp_keys, res_keys)

    def test_sparse_balance_basis_invalid(self):
        with self.assertRaises(ValueError):
            tree = u"(a,b,c);"
            t = TreeNode.read([tree])
            sparse_balance_basis(t)

    def test_sparse_balance_basis_unbalanced(self):
        tree = u"((a,b)c, d);"
        t = TreeNode.read([tree])
        exp_basis = coo_matrix(np.array(
            [[-np.sqrt(1. / 6), -np.sqrt(1. / 6), np.sqrt(2. / 3)],
             [-np.sqrt(1. / 2), np.sqrt(1. / 2), 0]]
        ))
        exp_keys = [t.name, t[0].name]
        res_basis, res_keys = sparse_balance_basis(t)

        assert_coo_allclose(exp_basis, res_basis)
        self.assertListEqual(exp_keys, res_keys)

    def test_sparse_balance_basis_unbalanced2(self):
        tree = u"(d, (a,b)c);"

        t = TreeNode.read([tree])

        exp_basis = coo_matrix(np.array(
            [
                [-np.sqrt(2. / 3), np.sqrt(1. / 6), np.sqrt(1. / 6)],
                [0, -np.sqrt(1. / 2), np.sqrt(1. / 2)]
            ]
        ))

        exp_keys = [t.name, t[1].name]
        res_basis, res_keys = sparse_balance_basis(t)
        assert_coo_allclose(exp_basis, res_basis, atol=1e-7, rtol=1e-7)
        self.assertListEqual(exp_keys, res_keys)


class TestBalances(unittest.TestCase):

    def test_count_matrix_base_case(self):
        tree = u"(a,b);"
        t = TreeNode.read([tree])
        res, _ = _count_matrix(t)
        exp = {'k': 0, 'l': 1, 'r': 1, 't': 0, 'tips': 2}
        self.assertEqual(res[t], exp)

        exp = {'k': 0, 'l': 0, 'r': 0, 't': 0, 'tips': 1}
        self.assertEqual(res[t[0]], exp)

        exp = {'k': 0, 'l': 0, 'r': 0, 't': 0, 'tips': 1}
        self.assertEqual(res[t[1]], exp)

    def test_count_matrix_unbalanced(self):
        tree = u"((a,b)c, d);"
        t = TreeNode.read([tree])
        res, _ = _count_matrix(t)

        exp = {'k': 0, 'l': 2, 'r': 1, 't': 0, 'tips': 3}
        self.assertEqual(res[t], exp)
        exp = {'k': 1, 'l': 1, 'r': 1, 't': 0, 'tips': 2}
        self.assertEqual(res[t[0]], exp)

        exp = {'k': 0, 'l': 0, 'r': 0, 't': 0, 'tips': 1}
        self.assertEqual(res[t[1]], exp)
        self.assertEqual(res[t[0][0]], exp)
        self.assertEqual(res[t[0][1]], exp)

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
        exp_keys = [t.name]
        res_basis, res_keys = _balance_basis(t)

        npt.assert_allclose(exp_basis, res_basis)
        self.assertListEqual(exp_keys, res_keys)

    def test__balance_basis_unbalanced(self):
        tree = u"((a,b)c, d);"
        t = TreeNode.read([tree])

        exp_basis = np.array(
            [[-np.sqrt(1. / 6), -np.sqrt(1. / 6), np.sqrt(2. / 3)],
             [-np.sqrt(1. / 2), np.sqrt(1. / 2), 0]]
        )
        exp_keys = [t.name, t[0].name]
        res_basis, res_keys = _balance_basis(t)

        npt.assert_allclose(exp_basis, res_basis)
        self.assertListEqual(exp_keys, res_keys)

    def test_balance_basis_base_case(self):
        tree = u"(a,b);"
        t = TreeNode.read([tree])
        exp_keys = [t.name]
        exp_basis = np.array([0.19557032, 0.80442968])
        res_basis, res_keys = balance_basis(t)

        npt.assert_allclose(exp_basis, res_basis)
        self.assertListEqual(exp_keys, res_keys)

    def test_balance_basis_unbalanced(self):
        tree = u"((a,b)c, d);"
        t = TreeNode.read([tree])
        exp_keys = [t.name, t[0].name]
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
        npt.assert_allclose(exp_basis[:, ::-1], res_basis)


if __name__ == "__main__":
    unittest.main()
