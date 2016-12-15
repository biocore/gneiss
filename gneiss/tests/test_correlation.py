# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the GPLv3 License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------
import numpy as np
import pandas as pd
import unittest
from gneiss.correlation import  lovell_distance
import pandas.util.testing as pdt
from skbio import DistanceMatrix
import numpy.testing as npt


class TestLovell(unittest.TestCase):
    def setUp(self):
        pass

    def test_lovell_distance(self):
        t = pd.DataFrame([[1, 1, 2, 3, 1, 4],
                          [2, 2, 0.1, 4, 1, .1],
                          [3, 3.1, 2, 3, 2, 2],
                          [4.1, 4, 0.2, 1, 1, 2.5]],
                         index=['S1', 'S2', 'S3', 'S4'],
                         columns=['F1', 'F2', 'F3', 'F4', 'F5', 'F6'])

        res_dm = lovell_distance(t)
        exp_dm = np.array(
            [[0.00000000e+00, 9.83970716e-04, 6.21481225e+00,
              2.14782516e+00, 6.25833109e-01, 5.73857550e+00],
             [1.01064057e-03, 0.00000000e+00, 6.29597813e+00,
              2.16577748e+00, 6.11550940e-01, 5.89028012e+00],
             [2.70438228e+00, 2.66740354e+00, 0.00000000e+00,
              1.84670514e+00, 1.48388016e+00, 1.09564058e+00],
             [1.62779731e+00, 1.59808796e+00, 3.21631732e+00,
              0.00000000e+00, 5.38392284e-01, 5.63421904e+00],
             [2.74323334e+00, 2.60989056e+00, 1.49473077e+01,
              3.11387834e+00, 0.00000000e+00, 2.08435300e+01],
             [1.97285811e+00, 1.97157428e+00, 8.65605074e-01,
              2.55578359e+00, 1.63477809e+00, 0.00000000e+00]])
        exp_dm = DistanceMatrix(0.5 * (exp_dm + exp_dm.T),
                                ids=['F1', 'F2', 'F3', 'F4', 'F5', 'F6'])
        npt.assert_allclose(exp_dm.data, res_dm.data)
        self.assertEqual(exp_dm.ids, res_dm.ids)

        # assert that this still works with an array
        res_dm = lovell_distance(t.values)
        npt.assert_allclose(exp_dm.data, res_dm.data)


if __name__== '__main__':
    unittest.main()
