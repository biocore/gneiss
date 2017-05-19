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
import pandas.util.testing as pdt
import os
import numpy.testing as npt


# create some mock classes for testing
class submock_ok(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def summary(self):
        print("OK!")

    def fit(self, **kwargs):
        """ Fit the model """
        for s in self.submodels:
            # assumes that the underlying submodels have implemented `fit`.
            m = s.fit(**kwargs)
            self.results.append(m)


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

    def tearDown(self):
        if os.path.exists(self.pickle_fname):
            os.remove(self.pickle_fname)

    def test_init(self):
        submodels = [None, None]

        res = submock_ok(submodels=submodels, balances=self.balances)
        # check submodels
        self.assertTrue(res.submodels[0] is None)
        self.assertTrue(res.submodels[1] is None)

        # check balances
        pdt.assert_frame_equal(self.balances, res.balances)
        # check results
        self.assertEqual(res.results, [])

    def test_bad_init(self):
        # makes sure that the summary() is implemented in the base class
        submodels = [None, None]

        with self.assertRaises(TypeError):
            submock_bad(submodels=submodels, balances=self.balances)

    # pickle io tests
    def test_read_write(self):

        # now initialize model
        submodels = [self.model1, self.model2]

        exp = submock_ok(submodels=submodels, balances=self.balances)

        exp.write_pickle(self.pickle_fname)

        res = submock_ok.read_pickle(self.pickle_fname)
        res.fit()
        exp1 = self.model1.fit()
        exp2 = self.model2.fit()

        # check balances
        pdt.assert_frame_equal(self.balances, res.balances)
        # check results
        npt.assert_allclose(res.results[0].predict(),
                            exp1.predict())
        npt.assert_allclose(res.results[1].predict(),
                            exp2.predict())

    def test_read_write_handle(self):
        submodels = [self.model1, self.model2]
        with open(self.pickle_fname, 'wb') as wfh:
            exp = submock_ok(submodels=submodels, balances=self.balances)

            exp.write_pickle(wfh)

        with open(self.pickle_fname, 'rb') as rfh:
            res = submock_ok.read_pickle(rfh)

        res.fit()
        exp1 = self.model1.fit()
        exp2 = self.model2.fit()

        # check balances
        pdt.assert_frame_equal(self.balances, res.balances)
        # check results
        npt.assert_allclose(res.results[0].predict(),
                            exp1.predict())
        npt.assert_allclose(res.results[1].predict(),
                            exp2.predict())


if __name__ == '__main__':
    unittest.main()
