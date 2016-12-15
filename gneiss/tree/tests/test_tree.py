# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the GPLv3 License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

import pandas as pd
import unittest
from gneiss.tree._pba import hcpba, supgma


class TestPBA(unittest.TestCase):
    def setUp(self):
        pass

    def test_hcpba_1(self):
        table = pd.DataFrame(
            [[1, 1, 0, 0, 0],
             [0, 1, 1, 0, 0],
             [0, 0, 1, 1, 0],
             [0, 0, 0, 1, 1]],
            columns=['s1', 's2', 's3', 's4', 's5'],
            index=['o1', 'o2', 'o3', 'o4']).T
        exp_str = ('((o1:0.935064935065,o2:0.935064935065):0.649350649351,'
                   '(o3:0.935064935065,o4:0.935064935065):0.649350649351);\n')
        res_tree = hcpba(table+0.1)
        self.assertEqual(exp_str, str(res_tree))

    def test_hcpba_2(self):
        t = pd.DataFrame([[1, 1, 2, 3, 1, 4],
                          [2, 2, 0.1, 4, 1, .1],
                          [3, 3.1, 2, 3, 2, 2],
                          [4.1, 4, 0.2, 1, 1, 2.5]],
                         index=['S1', 'S2', 'S3', 'S4'],
                         columns=['F1', 'F2', 'F3', 'F4', 'F5', 'F6'])
        exp_str = ('((F3:0.490311414185,F6:0.490311414185):2.18526312266,'
                   '(F4:0.932646544468,(F5:0.823813493441,'
                   '(F1:0.000498652822567,F2:0.000498652822567):'
                   '0.823314840619):0.108833051027):1.74292799238);\n')
        res_tree = hcpba(t)
        self.assertEqual(exp_str, str(res_tree))


class TestUPGMA(unittest.TestCase):
    def setUp(self):
        pass

    def test_supgma(self):
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
        res_tree = supgma(table, gradient)
        exp_str = '((o1:0.5,o2:0.5):0.5,(o3:0.5,o4:0.5):0.5);\n'
        self.assertEqual(exp_str, str(res_tree))


if __name__ == '__main__':
    unittest.main()
