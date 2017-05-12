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
from gneiss.cluster import random_linkage, gradient_linkage
import pandas.util.testing as pdt


class TestILRTransform(unittest.TestCase):

    def test_ilr(self):
        np.random.seed(0)
        table = pd.DataFrame([[1, 1, 2, 2],
                              [1, 2, 2, 1],
                              [2, 2, 1, 1]],
                             index=[1, 2, 3],
                             columns=['a', 'b', 'c', 'd'])
        table = table.reindex(columns=np.random.permutation(table.columns))
        ph = pd.Series([1, 2, 3], index=table.index)
        tree = gradient_linkage(table, ph)
        res_balances = ilr_transform(table, tree)
        exp_balances = pd.DataFrame(
            [[-0.693147, 5.551115e-17, -2.775558e-17],
             [0.000000, -4.901291e-01, -4.901291e-01],
             [0.693147, -5.551115e-17, 2.775558e-17]],
            columns=['y0', 'y1', 'y2'],
            index=[1, 2, 3])
        pdt.assert_frame_equal(res_balances, exp_balances)

    # using the unittest from skbio
    def test_basis(self):
        np.random.seed(0)
        N, D = 10, 10
        ids = np.arange(D).astype(np.str)
        sample_ids = np.arange(D).astype(np.str)
        table = pd.DataFrame(np.random.random((N, D)),
                             index=sample_ids, columns=ids)
        tree = random_linkage(D)

        res = ilr_transform(table, tree)

        exp = pd.DataFrame(
            {'y0': [-0.098623, -1.301295, -0.824148, -0.187292, -0.875373,
                    -1.321327, 0.158169, -1.841411, 1.084442, -1.018260],
             'y1': [0.146379, -0.300242, -0.389586, 0.660788, 0.350369,
                    0.117449, 0.429935, -0.299559, -0.545547, -0.058381],
             'y2': [0.308686, -0.026311, 0.473282, 2.259383, 0.445904,
                    0.914465, 0.709606, -1.902002, -0.627603, -1.853166],
             'y3': [-0.120931, 0.050495, -0.388288, -0.374096, 0.330703,
                    0.574503, 1.260615, 0.519287, -1.318978, -1.147023],
             'y4': [-0.177945, -1.815321, -1.334285, -2.410931, 1.700129,
                    0.506523, 0.249257, 0.141903, -0.141854, -0.640855],
             'y5': [-0.042076, 2.168831, 0.054688, 0.290070, 0.277161,
                    0.133657, -0.091957, 0.583960, 1.242143, -0.798105],
             'y6': [-0.027264, 0.325122, -0.451744, -0.004142, -1.146480,
                    0.755110, -0.217937, 1.264411, -0.146839, -0.426935],
             'y7': [-0.651637, 0.078898, -0.162573, -0.229876, 0.100711,
                    -0.456750, -1.531296, -0.646296, 0.338289, -3.658251],
             'y8': [0.503414, 2.629012, 1.333266, 0.005567, -0.346261,
                    -0.669773, -1.509504, -0.605026, -1.214381, -2.385142]},
            index=ids)

        pdt.assert_frame_equal(exp, res, check_less_precise=True)


if __name__ == '__main__':
    unittest.main()
