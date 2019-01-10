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
from gneiss.composition._composition import ilr_transform
import pandas.util.testing as pdt
from skbio import TreeNode


class TestILRTransform(unittest.TestCase):

    def test_ilr(self):
        np.random.seed(0)
        table = pd.DataFrame([[1, 1, 2, 2],
                              [1, 2, 2, 1],
                              [2, 2, 1, 1]],
                             index=[1, 2, 3],
                             columns=['a', 'b', 'c', 'd'])
        table = table.reindex(columns=np.random.permutation(table.columns))
        # fix tree
        tree = TreeNode.read([r"((d,c)y2, (b,a)y1)y0;"])
        res_balances = ilr_transform(table, tree)
        exp_balances = pd.DataFrame(
            [[0.693147, -5.551115e-17, 2.775558e-17],
             [0.000000, -4.901291e-01, -4.901291e-01],
             [-0.693147, 5.551115e-17, -2.775558e-17]],
            columns=['y0', 'y1', 'y2'],
            index=[1, 2, 3])
        exp_balances = exp_balances.reindex(columns=res_balances.columns)
        pdt.assert_frame_equal(res_balances, exp_balances)


if __name__ == '__main__':
    unittest.main()
