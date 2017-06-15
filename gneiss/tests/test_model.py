# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------
import pandas as pd
import statsmodels.formula.api as smf
from skbio import TreeNode
from gneiss._model import Model
import unittest
import os
import pandas.util.testing as pdt


# create some mock classes for testing
class submock_ok(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def summary(self):
        print("OK!")

    def fit(self, **kwargs):
        pass


class submock_bad(Model):
    def __init__(self, **kwargs):
        super(Model, self, **kwargs)


class TestModel(unittest.TestCase):
    def setUp(self):
        self.pickle_fname = "test.pickle"
        self.data = pd.DataFrame([[1, 1, 1],
                                  [3, 2, 3],
                                  [4, 3, 2],
                                  [5, 4, 4],
                                  [2, 5, 3],
                                  [3, 6, 5],
                                  [4, 7, 4]],
                                 index=['s1', 's2', 's3', 's4',
                                        's5', 's6', 's7'],
                                 columns=['Y1', 'Y2', 'X'])

        self.model1 = smf.ols(formula="Y1 ~ X", data=self.data)
        self.model2 = smf.ols(formula="Y2 ~ X", data=self.data)

        self.basis = pd.DataFrame([[0.80442968, 0.19557032]],
                                  index=['a'],
                                  columns=['x', 'y'])
        self.tree = TreeNode.read(['(x, y)a;'])
        self.balances = pd.DataFrame({'a': [-1, 0, 1]})
        self.metadata = pd.DataFrame(
            [[1], [3], [2]],
            columns=['X'])

    def tearDown(self):
        if os.path.exists(self.pickle_fname):
            os.remove(self.pickle_fname)

    def test_init(self):
        res = submock_ok(Y=self.balances, Xs=self.metadata)

        # check balances
        pdt.assert_frame_equal(self.balances, res.response_matrix)


if __name__ == '__main__':
    unittest.main()
