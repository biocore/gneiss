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
from skbio.tree import TreeNode
from gneiss.composition._composition import ilr_hierarchical, ilr_phylogenetic
from gneiss.cluster import gradient_linkage
import pandas.util.testing as pdt


class TestILRTransform(unittest.TestCase):

    def test_ilr_hierarchical(self):
        np.random.seed(0)
        table = pd.DataFrame([[1, 1, 2, 2],
                              [1, 2, 2, 1],
                              [2, 2, 1, 1]],
                             index=[1, 2, 3],
                             columns=['a', 'b', 'c', 'd'])
        table = table.reindex(columns=np.random.permutation(table.columns))
        ph = pd.Series([1, 2, 3], index=table.index)
        tree = gradient_linkage(table, ph)
        res_balances = ilr_hierarchical(table, tree)
        exp_balances = pd.DataFrame(
            [[0.693147, -5.551115e-17, 2.775558e-17],
             [0.000000, 4.901291e-01, -4.901291e-01],
             [-0.693147, 5.551115e-17, -2.775558e-17]],
            columns=['y0', 'y1', 'y2'],
            index=[1, 2, 3])
        pdt.assert_frame_equal(res_balances, exp_balances)

    def test_ilr_phylogenetic(self):
        np.random.seed(0)
        table = pd.DataFrame([[1, 1, 2, 2],
                              [1, 2, 2, 1],
                              [2, 2, 1, 1]],
                             index=[1, 2, 3],
                             columns=['a', 'b', 'c', 'd'])
        table = table.reindex(columns=np.random.permutation(table.columns))
        tree = TreeNode.read([
            '((c:0.025,d:0.025,e:0.025):0.2,(b:0.025,a:0.025):0.2);'])
        res_tree, res_balances = ilr_phylogenetic(table, tree)
        exp_tree_str = ('((e:0.025,(c:0.025,d:0.025)y3)y1:0.2,'
                        '(b:0.025,a:0.025)y2:0.2)y0;\n')
        self.assertEqual(str(res_tree), exp_tree_str)
        exp_balances = pd.DataFrame(
            [[-0.693147, 0.0, -2.775558e-17],
             [5.551115e-17, -4.901291e-01, 4.901291e-01],
             [0.693147, -5.551115e-17, 2.775558e-17]],
            columns=['y0', 'y2', 'y3'],
            index=[1, 2, 3])
        pdt.assert_frame_equal(res_balances, exp_balances)


if __name__ == '__main__':
    unittest.main()
