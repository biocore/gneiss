# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the GPLv3 License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

import pandas as pd
import unittest
from gneiss.cluster._pba import proportional_linkage, gradient_linkage


class TestPBA(unittest.TestCase):
    def setUp(self):
        pass

    def test_proportional_linkage_1(self):
        table = pd.DataFrame(
            [[1, 1, 0, 0, 0],
             [0, 1, 1, 0, 0],
             [0, 0, 1, 1, 0],
             [0, 0, 0, 1, 1]],
            columns=['s1', 's2', 's3', 's4', 's5'],
            index=['o1', 'o2', 'o3', 'o4']).T
        exp_str = ('((o1:0.574990173931,o2:0.574990173931):0.773481312844,'
                   '(o3:0.574990173931,o4:0.574990173931):0.773481312844);\n')
        res_tree = proportional_linkage(table+0.1)
        self.assertEqual(exp_str, str(res_tree))

    def test_proportional_linkage_2(self):
        t = pd.DataFrame([[1, 1, 2, 3, 1, 4],
                          [2, 2, 0.1, 4, 1, .1],
                          [3, 3.1, 2, 3, 2, 2],
                          [4.1, 4, 0.2, 1, 1, 2.5]],
                         index=['S1', 'S2', 'S3', 'S4'],
                         columns=['F1', 'F2', 'F3', 'F4', 'F5', 'F6'])
        exp_str = ('((F4:0.228723591874,(F5:0.074748541601,'
                   '(F1:0.00010428164962,F2:0.00010428164962):'
                   '0.0746442599513):0.153975050273):0.70266138894,'
                   '(F3:0.266841737789,F6:0.266841737789):0.664543243026);\n')
        res_tree = proportional_linkage(t)
        self.assertEqual(exp_str, str(res_tree))


class TestUPGMA(unittest.TestCase):
    def setUp(self):
        pass

    def test_gradient_linkage(self):
        table = pd.DataFrame(
            [[1, 1, 0, 0, 0],
             [0, 1, 10, 0, 0],
             [0, 0, 1, 1, 0],
             [0, 0, 0, 1, 1]],
            columns=['s1', 's2', 's3', 's4', 's5'],
            index=['o1', 'o2', 'o3', 'o4']).T
        gradient = pd.Series(
            [1, 2, 3, 4, 5],
            index=['s1', 's2', 's3', 's4', 's5'])
        res_tree = gradient_linkage(table, gradient)
        exp_str = '((o1:0.5,o2:0.5):0.5,(o3:0.5,o4:0.5):0.5);\n'
        self.assertEqual(exp_str, str(res_tree))


if __name__ == '__main__':
    unittest.main()
