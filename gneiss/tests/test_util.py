# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

import unittest
import pandas as pd
import pandas.util.testing as pdt
from skbio import TreeNode
from gneiss.util import match, match_tips, rename_tips


class TestUtil(unittest.TestCase):

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
        pdt.assert_frame_equal(exp_table, res_table)
        pdt.assert_frame_equal(exp_metadata, res_metadata)

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

        res_table, res_metadata = match(table, metadata, intersect=True)
        pdt.assert_frame_equal(exp_table, res_table)
        pdt.assert_frame_equal(exp_metadata, res_metadata)

    def test_match_mismatch(self):
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
        with self.assertRaises(ValueError):
            match(table, metadata)

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
        res_table, res_tree = match_tips(table, tree, intersect=True)
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
        res_table, res_tree = match_tips(table, tree, intersect=True)
        pdt.assert_frame_equal(exp_table, res_table)
        self.assertEqual(str(exp_tree), str(res_tree))

    def test_match_tips_mismatch(self):
        # table has less columns than tree tips
        table = pd.DataFrame([[0, 0, 1],
                              [2, 3, 4],
                              [5, 5, 3],
                              [0, 0, 1]],
                             index=['s1', 's2', 's3', 's4'],
                             columns=['a', 'b', 'd'])
        tree = TreeNode.read([u"(((a,b)f, c),d)r;"])
        with self.assertRaises(ValueError):
            match_tips(table, tree)

        table = pd.DataFrame([[0, 0, 1, 1],
                              [2, 3, 4, 4],
                              [5, 5, 3, 3],
                              [0, 0, 0, 1]],
                             index=['s1', 's2', 's3', 's4'],
                             columns=['a', 'b', 'c', 'd'])
        tree = TreeNode.read([u"((a,b)f,d)r;"])
        with self.assertRaises(ValueError):
            match_tips(table, tree)

    def test_rename_tips(self):
        tree = TreeNode.read([u"(((a,b), c),d)r;"])
        exp_tree = TreeNode.read([u"(((a,b)y2, c)y1,d)y0;"])
        res_tree = rename_tips(tree)
        self.assertEqual(str(exp_tree), str(res_tree))

    def test_rename_tips_names(self):
        tree = TreeNode.read([u"(((a,b), c),d)r;"])
        exp_tree = TreeNode.read([u"(((a,b)ab, c)abc,d)r;"])
        res_tree = rename_tips(tree, ['r', 'abc', 'ab'])
        self.assertEqual(str(exp_tree), str(res_tree))

    def test_rename_tips_names_mismatch(self):
        tree = TreeNode.read([u"(((a,b), c),d)r;"])
        with self.assertRaises(IndexError):
            rename_tips(tree, ['r', 'abc'])

if __name__ == '__main__':
    unittest.main()
