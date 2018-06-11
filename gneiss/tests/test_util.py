# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

import unittest
import numpy as np
import pandas as pd
import pandas.util.testing as pdt
from skbio import TreeNode
from gneiss.util import match, match_tips, design_formula
from gneiss.util import (rename_internal_nodes,
                         _type_cast_to_float, block_diagonal, band_diagonal,
                         split_balance, check_internal_nodes)
from biom import Table
from patsy import dmatrix
import numpy.testing as npt


class TestMatch(unittest.TestCase):

    def test_match(self):
        table = pd.DataFrame([[0, 0, 1, 1],
                              [2, 2, 4, 4],
                              [5, 5, 3, 3],
                              [0, 0, 0, 1]],
                             index=['s1', 's2', 's3', 's4'],
                             columns=['o1', 'o2', 'o3', 'o4'])
        metadata = pd.DataFrame([['a', 'control'],
                                 ['b', 'control'],
                                 ['c', 'diseased'],
                                 ['d', 'diseased']],
                                index=['s1', 's2', 's3', 's4'],
                                columns=['Barcode', 'Treatment'])
        exp_table, exp_metadata = table, metadata
        res_table, res_metadata = match(table, metadata)

        # make sure that the metadata and table indeces match
        pdt.assert_index_equal(res_table.index, res_metadata.index)

        res_table = res_table.sort_index()
        exp_table = exp_table.sort_index()

        res_metadata = res_metadata.sort_index()
        exp_metadata = exp_metadata.sort_index()

        pdt.assert_frame_equal(exp_table, res_table)
        pdt.assert_frame_equal(exp_metadata, res_metadata)

    def test_match_empty(self):
        table = pd.DataFrame([[0, 0, 1, 1],
                              [2, 2, 4, 4],
                              [5, 5, 3, 3],
                              [0, 0, 0, 1]],
                             index=['s1', 's2', 's3', 's4'],
                             columns=['o1', 'o2', 'o3', 'o4'])
        metadata = pd.DataFrame([['a', 'control'],
                                 ['b', 'control'],
                                 ['c', 'diseased'],
                                 ['d', 'diseased']],
                                index=['a1', 'a2', 'a3', 'a4'],
                                columns=['Barcode', 'Treatment'])

        with self.assertRaises(ValueError):
            match(table, metadata)

    def test_match_immutable(self):
        # tests to make sure that the original tables don't change.
        table = pd.DataFrame([[0, 0, 1, 1],
                              [2, 2, 4, 4],
                              [5, 5, 3, 3],
                              [0, 0, 0, 1]],
                             index=['s1', 's2', 's3', 's4'],
                             columns=['o1', 'o2', 'o3', 'o4'])
        metadata = pd.DataFrame([['a', 'control'],
                                 ['c', 'diseased'],
                                 ['b', 'control']],
                                index=['s1', 's3', 's2'],
                                columns=['Barcode', 'Treatment'])

        exp_table = pd.DataFrame([[0, 0, 1, 1],
                                  [2, 2, 4, 4],
                                  [5, 5, 3, 3],
                                  [0, 0, 0, 1]],
                                 index=['s1', 's2', 's3', 's4'],
                                 columns=['o1', 'o2', 'o3', 'o4'])
        exp_metadata = pd.DataFrame([['a', 'control'],
                                     ['c', 'diseased'],
                                     ['b', 'control']],
                                    index=['s1', 's3', 's2'],
                                    columns=['Barcode', 'Treatment'])
        match(table, metadata)
        pdt.assert_frame_equal(table, exp_table)
        pdt.assert_frame_equal(metadata, exp_metadata)

    def test_match_duplicate(self):
        table1 = pd.DataFrame([[0, 0, 1, 1],
                               [2, 2, 4, 4],
                               [5, 5, 3, 3],
                               [0, 0, 0, 1]],
                              index=['s2', 's2', 's3', 's4'],
                              columns=['o1', 'o2', 'o3', 'o4'])
        metadata1 = pd.DataFrame([['a', 'control'],
                                  ['b', 'control'],
                                  ['c', 'diseased'],
                                  ['d', 'diseased']],
                                 index=['s1', 's2', 's3', 's4'],
                                 columns=['Barcode', 'Treatment'])

        table2 = pd.DataFrame([[0, 0, 1, 1],
                               [2, 2, 4, 4],
                               [5, 5, 3, 3],
                               [0, 0, 0, 1]],
                              index=['s1', 's2', 's3', 's4'],
                              columns=['o1', 'o2', 'o3', 'o4'])
        metadata2 = pd.DataFrame([['a', 'control'],
                                  ['b', 'control'],
                                  ['c', 'diseased'],
                                  ['d', 'diseased']],
                                 index=['s1', 's1', 's3', 's4'],
                                 columns=['Barcode', 'Treatment'])

        with self.assertRaises(ValueError):
            match(table1, metadata1)
        with self.assertRaises(ValueError):
            match(table2, metadata2)

    def test_match_scrambled(self):
        table = pd.DataFrame([[0, 0, 1, 1],
                              [2, 2, 4, 4],
                              [5, 5, 3, 3],
                              [0, 0, 0, 1]],
                             index=['s1', 's2', 's3', 's4'],
                             columns=['o1', 'o2', 'o3', 'o4'])
        metadata = pd.DataFrame([['a', 'control'],
                                 ['c', 'diseased'],
                                 ['b', 'control'],
                                 ['d', 'diseased']],
                                index=['s1', 's3', 's2', 's4'],
                                columns=['Barcode', 'Treatment'])
        exp_table = table
        exp_metadata = pd.DataFrame([['a', 'control'],
                                     ['b', 'control'],
                                     ['c', 'diseased'],
                                     ['d', 'diseased']],
                                    index=['s1', 's2', 's3', 's4'],
                                    columns=['Barcode', 'Treatment'])

        res_table, res_metadata = match(table, metadata)
        # make sure that the metadata and table indeces match
        pdt.assert_index_equal(res_table.index, res_metadata.index)

        res_table = res_table.sort_index()
        exp_table = exp_table.sort_index()

        res_metadata = res_metadata.sort_index()
        exp_metadata = exp_metadata.sort_index()

        pdt.assert_frame_equal(exp_table, res_table)
        pdt.assert_frame_equal(exp_metadata, res_metadata)

    def test_match_intersect(self):
        table = pd.DataFrame([[0, 0, 1, 1],
                              [2, 2, 4, 4],
                              [5, 5, 3, 3],
                              [0, 0, 0, 1]],
                             index=['s1', 's2', 's3', 's4'],
                             columns=['o1', 'o2', 'o3', 'o4'])
        metadata = pd.DataFrame([['a', 'control'],
                                 ['c', 'diseased'],
                                 ['b', 'control']],
                                index=['s1', 's3', 's2'],
                                columns=['Barcode', 'Treatment'])

        exp_table = pd.DataFrame([[0, 0, 1, 1],
                                  [2, 2, 4, 4],
                                  [5, 5, 3, 3]],
                                 index=['s1', 's2', 's3'],
                                 columns=['o1', 'o2', 'o3', 'o4'])

        exp_metadata = pd.DataFrame([['a', 'control'],
                                     ['b', 'control'],
                                     ['c', 'diseased']],
                                    index=['s1', 's2', 's3'],
                                    columns=['Barcode', 'Treatment'])

        res_table, res_metadata = match(table, metadata)
        # sort for comparison, since the match function
        # scrambles the names due to hashing.
        res_table = res_table.sort_index()
        res_metadata = res_metadata.sort_index()
        pdt.assert_frame_equal(exp_table, res_table)
        pdt.assert_frame_equal(exp_metadata, res_metadata)

    def test_match_tips(self):
        table = pd.DataFrame([[0, 0, 1, 1],
                              [2, 2, 4, 4],
                              [5, 5, 3, 3],
                              [0, 0, 0, 1]],
                             index=['s1', 's2', 's3', 's4'],
                             columns=['a', 'b', 'c', 'd'])
        tree = TreeNode.read([u"(((a,b)f, c),d)r;"])
        exp_table, exp_tree = table, tree
        res_table, res_tree = match_tips(table, tree)
        pdt.assert_frame_equal(exp_table, res_table)
        self.assertEqual(str(exp_tree), str(res_tree))

    def test_match_tips_scrambled_tips(self):
        table = pd.DataFrame([[0, 0, 1, 1],
                              [2, 3, 4, 4],
                              [5, 5, 3, 3],
                              [0, 0, 0, 1]],
                             index=['s1', 's2', 's3', 's4'],
                             columns=['a', 'b', 'c', 'd'])
        tree = TreeNode.read([u"(((b,a)f, c),d)r;"])
        exp_tree = tree
        exp_table = pd.DataFrame([[0, 0, 1, 1],
                                  [3, 2, 4, 4],
                                  [5, 5, 3, 3],
                                  [0, 0, 0, 1]],
                                 index=['s1', 's2', 's3', 's4'],
                                 columns=['b', 'a', 'c', 'd'])

        res_table, res_tree = match_tips(table, tree)
        pdt.assert_frame_equal(exp_table, res_table)
        self.assertEqual(str(exp_tree), str(res_tree))

    def test_match_tips_scrambled_columns(self):
        table = pd.DataFrame([[0, 0, 1, 1],
                              [3, 2, 4, 4],
                              [5, 5, 3, 3],
                              [0, 0, 0, 1]],
                             index=['s1', 's2', 's3', 's4'],
                             columns=['b', 'a', 'c', 'd'])
        tree = TreeNode.read([u"(((a,b)f, c),d)r;"])
        exp_tree = tree
        exp_table = pd.DataFrame([[0, 0, 1, 1],
                                  [2, 3, 4, 4],
                                  [5, 5, 3, 3],
                                  [0, 0, 0, 1]],
                                 index=['s1', 's2', 's3', 's4'],
                                 columns=['a', 'b', 'c', 'd'])

        res_table, res_tree = match_tips(table, tree)
        pdt.assert_frame_equal(exp_table, res_table)
        self.assertEqual(str(exp_tree), str(res_tree))

    def test_match_tips_intersect_tips(self):
        # there are less tree tips than table columns
        table = pd.DataFrame([[0, 0, 1, 1],
                              [2, 3, 4, 4],
                              [5, 5, 3, 3],
                              [0, 0, 0, 1]],
                             index=['s1', 's2', 's3', 's4'],
                             columns=['a', 'b', 'c', 'd'])
        tree = TreeNode.read([u"((a,b)f,d)r;"])
        exp_table = pd.DataFrame([[0, 0, 1],
                                  [2, 3, 4],
                                  [5, 5, 3],
                                  [0, 0, 1]],
                                 index=['s1', 's2', 's3', 's4'],
                                 columns=['a', 'b', 'd'])
        exp_tree = tree
        res_table, res_tree = match_tips(table, tree)
        pdt.assert_frame_equal(exp_table, res_table)
        self.assertEqual(str(exp_tree), str(res_tree))

    def test_match_tips_intersect_columns(self):
        # table has less columns than tree tips
        table = pd.DataFrame([[0, 0, 1],
                              [2, 3, 4],
                              [5, 5, 3],
                              [0, 0, 1]],
                             index=['s1', 's2', 's3', 's4'],
                             columns=['a', 'b', 'd'])
        tree = TreeNode.read([u"(((a,b)f, c),d)r;"])
        exp_table = pd.DataFrame([[1, 0, 0],
                                  [4, 2, 3],
                                  [3, 5, 5],
                                  [1, 0, 0]],
                                 index=['s1', 's2', 's3', 's4'],
                                 columns=['d', 'a', 'b'])
        exp_tree = TreeNode.read([u"(d,(a,b)f)r;"])
        res_table, res_tree = match_tips(table, tree)
        pdt.assert_frame_equal(exp_table, res_table)
        self.assertEqual(str(exp_tree), str(res_tree))

    def test_match_tips_intersect_tree_immutable(self):
        # tests to see if tree changes.
        table = pd.DataFrame([[0, 0, 1],
                              [2, 3, 4],
                              [5, 5, 3],
                              [0, 0, 1]],
                             index=['s1', 's2', 's3', 's4'],
                             columns=['a', 'b', 'd'])
        tree = TreeNode.read([u"(((a,b)f, c),d)r;"])
        match_tips(table, tree)
        self.assertEqual(str(tree), u"(((a,b)f,c),d)r;\n")

    def test_biom_match(self):
        table = Table(
            np.array([[0, 0, 1, 1],
                      [2, 3, 4, 4],
                      [5, 5, 3, 3]]).T,
            ['a', 'b', 'c', 'd'],
            ['s2', 's3', 's4'])
        md = pd.DataFrame(
            {
                'x1': [1, 3, 2],
                'x2': [1, 1, 0]
            },
            columns=['s1', 's2', 's3']
        ).T

        exp_table = Table(
            np.array(
                [
                    [0, 0, 1, 1],
                    [2, 3, 4, 4]
                ]).T,
            ['a', 'b', 'c', 'd'],
            ['s2', 's3'])
        exp_md = pd.DataFrame(
            {
                'x1': [3, 2],
                'x2': [1, 0]
            },
            columns=['s2', 's3']
        ).T

        res_table, res_md = match(table, md)
        exp_df = pd.DataFrame(exp_table.to_dataframe())
        res_df = pd.DataFrame(res_table.to_dataframe())

        exp_df = exp_df.reindex_axis(sorted(exp_df.columns), axis=1)
        res_df = res_df.reindex_axis(sorted(res_df.columns), axis=1)

        pdt.assert_frame_equal(exp_df, res_df)

        exp_md = exp_md.reindex_axis(sorted(exp_md.index), axis=0)
        res_md = res_md.reindex_axis(sorted(res_md.index), axis=0)

        pdt.assert_frame_equal(res_md, exp_md)

    def test_biom_match_duplicate_md_error(self):
        table = Table(
            np.array([[0, 0, 1, 1],
                      [2, 3, 4, 4],
                      [5, 5, 3, 3]]).T,
            ['a', 'b', 'c', 'd'],
            ['s2', 's3', 's4'])
        md = pd.DataFrame(
            {
                'x1': [1, 3, 2],
                'x2': [1, 1, 0]
            },
            columns=['s2', 's2', 's3']
        ).T
        with self.assertRaises(ValueError):
            match(table, md)

    def test_biom_match_no_common_ids(self):
        table = Table(
            np.array([[0, 0, 1, 1],
                      [2, 3, 4, 4],
                      [5, 5, 3, 3]]).T,
            ['a', 'b', 'c', 'd'],
            ['y2', 'y3', 'y4'])
        md = pd.DataFrame(
            {
                'x1': [1, 3, 2],
                'x2': [1, 1, 0]
            },
            columns=['s2', 's2', 's3']
        ).T
        with self.assertRaises(ValueError):
            match(table, md)

    def test_biom_match_tips_intersect_tips(self):
        # there are less tree tips than table columns
        table = Table(
            np.array([[0, 0, 1, 1],
                      [2, 3, 4, 4],
                      [5, 5, 3, 3],
                      [0, 0, 0, 1]]).T,
            ['a', 'b', 'c', 'd'],
            ['s1', 's2', 's3', 's4'])

        tree = TreeNode.read([u"((a,b)f,c)r;"])
        exp_table = Table(
            np.array([[0, 0, 1],
                      [2, 3, 4],
                      [5, 5, 3],
                      [0, 0, 0]]).T,
            ['a', 'b', 'c'],
            ['s1', 's2', 's3', 's4'])

        exp_tree = tree
        res_table, res_tree = match_tips(table, tree)
        self.assertEqual(exp_table, res_table)
        self.assertEqual(str(exp_tree), str(res_tree))

    def test_biom_match_tips_intersect_columns(self):
        # table has less columns than tree tips
        table = Table(
            np.array([[0, 0, 1],
                      [2, 3, 4],
                      [5, 5, 3],
                      [0, 0, 1]]).T,
            ['a', 'b', 'd'],
            ['s1', 's2', 's3', 's4'])

        tree = TreeNode.read([u"(((a,b)f, c),d)r;"])
        table = Table(
            np.array([[0, 0, 1],
                      [2, 3, 4],
                      [5, 5, 3],
                      [0, 0, 1]]).T,
            ['a', 'b', 'd'],
            ['s1', 's2', 's3', 's4'])

        exp_table = Table(
            np.array([[1, 0, 0],
                      [4, 2, 3],
                      [3, 5, 5],
                      [1, 0, 0]]).T,
            ['d', 'a', 'b'],
            ['s1', 's2', 's3', 's4'])

        exp_tree = TreeNode.read([u"(d,(a,b)f)r;"])
        res_table, res_tree = match_tips(table, tree)
        self.assertEqual(exp_table, res_table)
        self.assertEqual(str(exp_tree), str(res_tree))

    def test_biom_match_tips_intersect_tree_immutable(self):
        # tests to see if tree changes.
        table = Table(
            np.array([[0, 0, 1],
                      [2, 3, 4],
                      [5, 5, 3],
                      [0, 0, 1]]).T,
            ['a', 'b', 'd'],
            ['s1', 's2', 's3', 's4'])

        exp_table = Table(
            np.array([[0, 0, 1],
                      [2, 3, 4],
                      [5, 5, 3],
                      [0, 0, 1]]).T,
            ['a', 'b', 'd'],
            ['s1', 's2', 's3', 's4'])

        tree = TreeNode.read([u"(((a,b)f, c),d)r;"])
        match_tips(table, tree)
        self.assertEqual(exp_table, table)
        self.assertEqual(str(tree), u"(((a,b)f,c),d)r;\n")

    def test_formula(self):
        train_metadata = pd.DataFrame(
            [['a', '1', 'control'],
             ['b', '2', 'control'],
             ['c', '1', 'diseased'],
             ['d', '3', 'diseased']],
            index=['s1', 's2', 's3', 's4'],
            columns=['Barcode', 'Group', 'Treatment'])

        test_metadata = pd.DataFrame(
            [['a', '1', 'control'],
             ['b', '1', 'control'],
             ['c', '1', 'diseased'],
             ['d', '1', 'diseased']],
            index=['s5', 's6', 's7', 's8'],
            columns=['Barcode', 'Group', 'Treatment'])

        exp_metadata = pd.DataFrame(
            [['a', 'control', 0],
             ['b', 'control', 0],
             ['c', 'diseased', 0],
             ['d', 'diseased', 0]],
            index=['s5', 's6', 's7', 's8'],
            columns=['Barcode', 'Treatment', 'Group'])

        formula = "C(Group) + C(Treatment)"
        exp_design = dmatrix(formula, exp_metadata,
                             return_type='dataframe')
        exp_design = pd.DataFrame(
            [
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 1.0]
            ], columns=['Intercept', 'C(Group)[T.2]',
                        'C(Group)[T.3]', 'C(Treatment)[T.diseased]'],
            index=['s5', 's6', 's7', 's8'],
        )

        res_design = design_formula(train_metadata, test_metadata, formula)[1]

        pdt.assert_frame_equal(exp_design, res_design)


class TestUtil(unittest.TestCase):

    def test_rename_internal_nodes(self):
        tree = TreeNode.read([u"(((a,b), c),d)r;"])
        exp_tree = TreeNode.read([u"(((a,b)y2, c)y1,d)y0;"])
        res_tree = rename_internal_nodes(tree)
        self.assertEqual(str(exp_tree), str(res_tree))

    def test_rename_internal_nodes_names(self):
        tree = TreeNode.read([u"(((a,b), c),d)r;"])
        exp_tree = TreeNode.read([u"(((a,b)ab, c)abc,d)r;"])
        res_tree = rename_internal_nodes(tree, ['r', 'abc', 'ab'])
        self.assertEqual(str(exp_tree), str(res_tree))

    def test_rename_internal_nodes_names_mismatch(self):
        tree = TreeNode.read([u"(((a,b), c),d)r;"])
        with self.assertRaises(ValueError):
            rename_internal_nodes(tree, ['r', 'abc'])

    def test_rename_internal_nodes_immutable(self):
        tree = TreeNode.read([u"(((a,b)y2, c),d)r;"])
        rename_internal_nodes(tree)
        self.assertEqual(str(tree), "(((a,b)y2,c),d)r;\n")

    def test_rename_internal_nodes_mutable(self):
        tree = TreeNode.read([u"(((a,b)y2, c),d)r;"])
        rename_internal_nodes(tree, inplace=True)
        self.assertEqual(str(tree), "(((a,b)y2,c)y1,d)y0;\n")

    def test_check_internal_nodes_error(self):
        tree = TreeNode.read([u"(((a,b)y2, c),d)r;"])
        with self.assertRaises(ValueError):
            check_internal_nodes(tree)

    def test_check_internal_nodes(self):
        tree = TreeNode.read([u"(((a,b)y2, c)x,d)r;"])
        check_internal_nodes(tree)

    def test_type_cast_to_float(self):
        x = pd.DataFrame({'a': [1, 2, 3, 4, 5],
                          'b': ['1', '2', '3', '4', '5'],
                          'c': ['a', 'b', 'c', 'd', 'e'],
                          'd': [1., 2., 3., 4., 5.]})
        res = _type_cast_to_float(x)
        exp = pd.DataFrame({'a': [1., 2., 3., 4., 5.],
                            'b': [1., 2., 3., 4., 5.],
                            'c': ['a', 'b', 'c', 'd', 'e'],
                            'd': [1., 2., 3., 4., 5.]})
        pdt.assert_frame_equal(res, exp)

    def test_block_diagonal_4x4(self):
        np.random.seed(0)
        res = block_diagonal(4, 4, 2)
        exp = np.array([[0.5488135, 0.71518937, 0., 0.],
                        [0.60276338, 0.54488318, 0., 0.],
                        [0., 0., 0.4236548, 0.64589411],
                        [0., 0., 0.43758721, 0.891773]])
        npt.assert_allclose(res, exp, rtol=1e-5, atol=1e-5)

    def test_block_diagonal_3x4(self):
        np.random.seed(0)
        res = block_diagonal(3, 4, 2)
        exp = np.array([[0.548814, 0., 0.],
                        [0.715189, 0., 0.],
                        [0., 0.602763, 0.544883],
                        [0., 0.423655, 0.645894]])

        npt.assert_allclose(res, exp, rtol=1e-5, atol=1e-5)

    def test_block_diagonal_error(self):

        with self.assertRaises(ValueError):
            block_diagonal(3, 4, 1)

        with self.assertRaises(ValueError):
            block_diagonal(3, 4, 0)

    def test_band_diagonal(self):
        res = band_diagonal(8, 3)
        exp = np.array([[0.33333333, 0., 0., 0., 0., 0.],
                        [0.33333333, 0.33333333, 0., 0., 0., 0.],
                        [0.33333333, 0.33333333, 0.33333333, 0., 0., 0.],
                        [0., 0.33333333, 0.33333333, 0.33333333, 0., 0.],
                        [0., 0., 0.33333333, 0.33333333, 0.33333333, 0.],
                        [0., 0., 0., 0.33333333, 0.33333333, 0.33333333],
                        [0., 0., 0., 0., 0.33333333, 0.33333333],
                        [0., 0., 0., 0., 0., 0.33333333]])
        npt.assert_allclose(res, exp, rtol=1e-4, atol=1e-4)


class TestSplitBalance(unittest.TestCase):

    def setUp(self):
        self.tree = TreeNode.read(['(x, y)a;'])
        self.balance = pd.Series([-1, 0, 1])
        self.balance.name = 'a'

    def test_split_balance(self):
        exp = pd.DataFrame([[0.19557032, 0.80442968],
                            [0.5, 0.5],
                            [0.80442968, 0.19557032]],
                           columns=['x', 'y'])
        pdt.assert_frame_equal(exp, split_balance(self.balance, self.tree))


if __name__ == '__main__':
    unittest.main()
