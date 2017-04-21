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
from skbio import DistanceMatrix
from skbio.util import get_data_path
from gneiss.composition._variance import variation_matrix


class TestVariationMatrix(unittest.TestCase):
    def setUp(self):
        pass

    def test_varmat1(self):
        X = pd.DataFrame({'x': np.arange(1, 10),
                          'y': np.arange(2, 11)})
        res = variation_matrix(X)
        exp = DistanceMatrix([[0, 0.032013010420979787 / 2],
                              [0.032013010420979787 / 2, 0]], ids=['x', 'y'])
        self.assertEqual(str(res), str(exp))

    def test_varmat_larg(self):
        np.random.seed(123)
        D = 50
        N = 100
        mean = np.ones(D)*10
        cov = np.eye(D)
        X = pd.DataFrame(np.abs(np.random.multivariate_normal(mean, cov,
                                                              size=N)),
                         columns=np.arange(D).astype(np.str))
        res = variation_matrix(X)

        exp = DistanceMatrix.read(get_data_path('exp_varmat.txt'))
        self.assertEqual(str(res), str(exp))


if __name__ == '__main__':
    unittest.main()
