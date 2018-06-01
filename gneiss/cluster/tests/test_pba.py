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
from gneiss.cluster._pba import (correlation_linkage, gradient_linkage,
                                 rank_linkage, random_linkage)
from skbio import TreeNode


class TestPBA(unittest.TestCase):
    def setUp(self):
        pass

    def test_correlation_linkage_1(self):
        table = pd.DataFrame(
            [[1, 1, 0, 0, 0],
             [0, 1, 1, 0, 0],
             [0, 0, 1, 1, 0],
             [0, 0, 0, 1, 1]],
            columns=['s1', 's2', 's3', 's4', 's5'],
            index=['o1', 'o2', 'o3', 'o4']).T
        exp_str = ('((o1:0.574990173931,o2:0.574990173931)y1:0.773481312844,'
                   '(o3:0.574990173931,o4:0.574990173931)y2:0.773481312844)'
                   'y0;\n')
        exp_tree = TreeNode.read([exp_str])
        res_tree = correlation_linkage(table+0.1)
        # only check for tree topology since checking for floating point
        # numbers on the branches is still tricky.
        self.assertEqual(exp_tree.ascii_art(), res_tree.ascii_art())

    def test_correlation_linkage_2(self):
        t = pd.DataFrame([[1, 1, 2, 3, 1, 4],
                          [2, 2, 0.1, 4, 1, .1],
                          [3, 3.1, 2, 3, 2, 2],
                          [4.1, 4, 0.2, 1, 1, 2.5]],
                         index=['S1', 'S2', 'S3', 'S4'],
                         columns=['F1', 'F2', 'F3', 'F4', 'F5', 'F6'])
        exp_str = ('((F4:0.228723591874,(F5:0.074748541601,'
                   '(F1:0.00010428164962,F2:0.00010428164962)'
                   'y4:0.0746442599513)y3:0.153975050273)'
                   'y1:0.70266138894,(F3:0.266841737789,F6:0.266841737789)'
                   'y2:0.664543243026)y0;\n')
        exp_tree = TreeNode.read([exp_str])
        res_tree = correlation_linkage(t)
        self.assertEqual(exp_tree.ascii_art(), res_tree.ascii_art())


class TestUPGMA(unittest.TestCase):
    def setUp(self):
        pass

    def test_gradient_linkage(self):
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
        res_tree = gradient_linkage(table, gradient)
        exp_str = '((o1:0.5,o2:0.5)y1:0.5,(o3:0.5,o4:0.5)y2:0.5)y0;\n'
        self.assertEqual(exp_str, str(res_tree))


class TestRandomLinkage(unittest.TestCase):

    def test_random_tree(self):
        np.random.seed(0)
        t = random_linkage(10)
        exp_str = (
            '((7:0.0359448798595,8:0.0359448798595)y1:0.15902486847,'
            '((9:0.0235897432375,(4:0.00696620596189,6:0.00696620596189)'
            'y5:0.0166235372756)y3:0.0747173561014,(1:0.0648004111784,'
            '((0:0.00196516046521,3:0.00196516046521)y7:0.0367750400883,'
            '(2:0.0215653684975,5:0.0215653684975)y8:0.017174832056)'
            'y6:0.0260602106249)y4:0.0335066881605)y2:0.0966626489905)y0;\n')
        exp_tree = TreeNode.read([exp_str])
        self.assertEqual(t.ascii_art(), exp_tree.ascii_art())


class TestRankLinkage(unittest.TestCase):

    def test_rank_linkage(self):
        ranks = pd.Series([1, 2, 4, 5],
                          index=['o1', 'o2', 'o3', 'o4'])
        t = rank_linkage(ranks)
        exp = '((o1:0.5,o2:0.5)y1:1.0,(o3:0.5,o4:0.5)y2:1.0)y0;\n'
        self.assertEqual(str(t), exp)


if __name__ == '__main__':
    unittest.main()
