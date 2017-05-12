# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------
import pandas as pd
from skbio import TreeNode
import numpy as np
from skbio.stats.composition import ilr_inv, clr_inv
import statsmodels.formula.api as smf
import pandas.util.testing as pdt
from gneiss.regression._model import RegressionModel
from gneiss.balances import balance_basis
import unittest
import os


# create some mock classes for testing
class submock(RegressionModel):
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


class TestRegressionModel(unittest.TestCase):
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
        self.tree = TreeNode.read(['((a,b)Y1, c)Y2;'])
        self.basis = pd.DataFrame(clr_inv(balance_basis(self.tree)[0]),
                                  columns=['a', 'b', 'c'],
                                  index=['Y1', 'Y2'])
        self.balances = pd.DataFrame(self.data[['Y1', 'Y2']],
                                     index=self.data.index,
                                     columns=['Y1', 'Y2'])

    def tearDown(self):
        if os.path.exists(self.pickle_fname):
            os.remove(self.pickle_fname)

    def test_regression_results_pvalues(self):
        # checks to see if pvalues are calculated correctly.

        submodels = [self.model1, self.model2]
        res = submock(submodels=submodels, balances=self.balances)
        res.fit()
        exp = pd.DataFrame({'Intercept': [0.307081, 0.972395],
                            'X': [0.211391, 0.029677]},
                           index=['Y1', 'Y2'])
        pdt.assert_frame_equal(res.pvalues, exp,
                               check_exact=False,
                               check_less_precise=True)

    def test_regression_results_coefficient(self):
        exp_coef = pd.DataFrame({'Intercept': [1.447368, -0.052632],
                                 'X': [0.539474, 1.289474]},
                                index=['Y1', 'Y2'])
        submodels = [self.model1, self.model2]
        res = submock(submodels=submodels, balances=self.balances)
        res.fit()
        pdt.assert_frame_equal(res.coefficients(), exp_coef,
                               check_exact=False,
                               check_less_precise=True)

    def test_regression_results_coefficient_projection(self):
        tree = TreeNode.read([r'(c, (a, b)Y2)Y1;'])
        exp_coef = pd.DataFrame(
            np.array([[0.47802399, 0.44373548, 0.07824052],
                      [0.11793186, 0.73047731, 0.15159083]]).T,
            columns=['Intercept', 'X'],
            index=['a', 'b', 'c'])

        submodels = [self.model1, self.model2]
        res = submock(submodels=submodels, balances=self.balances)
        res.fit()
        res_coef = res.coefficients(tree)
        res_coef = res_coef.sort_index()
        pdt.assert_frame_equal(res_coef, exp_coef,
                               check_exact=False,
                               check_less_precise=True)

    def test_regression_results_residuals_projection(self):
        tree = TreeNode.read([r'(c, (a, b)Y2)Y1;'])
        basis, _ = balance_basis(tree)
        exp_resid = pd.DataFrame({'s1': [-0.986842, -0.236842],
                                  's2': [-0.065789, -1.815789],
                                  's3': [1.473684, 0.473684],
                                  's4': [1.394737, -1.105263],
                                  's5': [-1.065789, 1.184211],
                                  's6': [-1.144737, -0.394737],
                                  's7': [0.394737, 1.894737]},
                                 index=['Y1', 'Y2']).T
        exp_resid = pd.DataFrame(ilr_inv(exp_resid, basis),
                                 index=['s1', 's2', 's3', 's4',
                                        's5', 's6', 's7'],
                                 columns=['c', 'a', 'b'])

        submodels = [self.model1, self.model2]
        res = submock(submodels=submodels, balances=self.balances)
        res.fit()
        res_resid = res.residuals(tree).sort_index()
        pdt.assert_frame_equal(res_resid, exp_resid,
                               check_exact=False,
                               check_less_precise=True)

    def test_regression_results_residuals(self):
        exp_resid = pd.DataFrame({'s1': [-0.986842, -0.236842],
                                  's2': [-0.065789, -1.815789],
                                  's3': [1.473684, 0.473684],
                                  's4': [1.394737, -1.105263],
                                  's5': [-1.065789, 1.184211],
                                  's6': [-1.144737, -0.394737],
                                  's7': [0.394737, 1.894737]},
                                 index=['Y1', 'Y2']).T
        submodels = [self.model1, self.model2]
        res = submock(submodels=submodels, balances=self.balances)
        res.fit()

        pdt.assert_frame_equal(res.residuals(), exp_resid,
                               check_exact=False,
                               check_less_precise=True)

    def test_regression_results_predict(self):
        submodels = [self.model1, self.model2]
        res = submock(submodels=submodels, balances=self.balances)
        res.fit()

        res_predict = res.predict(self.data[['X']])

        exp_predict = pd.DataFrame({'s1': [1.986842, 1.236842],
                                    's2': [3.065789, 3.815789],
                                    's3': [2.526316, 2.526316],
                                    's4': [3.605263, 5.105263],
                                    's5': [3.065789, 3.815789],
                                    's6': [4.144737, 6.394737],
                                    's7': [3.605263, 5.105263]},
                                   index=['Y1', 'Y2']).T

        pdt.assert_frame_equal(res_predict, exp_predict)

    def test_regression_results_predict_extrapolate(self):

        submodels = [self.model1, self.model2]
        res = submock(submodels=submodels, balances=self.balances)
        res.fit()

        extrapolate = pd.DataFrame({'X': [8, 9, 10]},
                                   index=['k1', 'k2', 'k3'])
        res_predict = res.predict(extrapolate)

        exp_predict = pd.DataFrame({'k1': [5.76315789, 10.26315789],
                                    'k2': [6.30263158, 11.55263158],
                                    'k3': [6.84210526, 12.84210526]},
                                   index=['Y1', 'Y2']).T

        pdt.assert_frame_equal(res_predict, exp_predict)

    def test_regression_results_predict_projection(self):
        submodels = [self.model1, self.model2]
        tree = TreeNode.read([r'(c, (a, b)Y2)Y1;'])
        basis, _ = balance_basis(tree)

        res = submock(submodels=submodels, balances=self.balances)
        res.fit()

        res_predict = res.predict(self.data[['X']], tree)
        exp_predict = [[1.986842, 1.236842],
                       [3.065789, 3.815789],
                       [2.526316, 2.526316],
                       [3.605263, 5.105263],
                       [3.065789, 3.815789],
                       [4.144737, 6.394737],
                       [3.605263, 5.105263]]

        exp_predict = pd.DataFrame(ilr_inv(exp_predict, basis),
                                   columns=['c', 'a', 'b'],
                                   index=['s1', 's2', 's3', 's4',
                                          's5', 's6', 's7'])
        pdt.assert_frame_equal(res_predict, exp_predict)

    def test_regression_results_predict_none(self):
        submodels = [self.model1, self.model2]
        res = submock(submodels=submodels, balances=self.balances)
        res.fit()
        res_predict = res.predict()

        exp_predict = pd.DataFrame({'s1': [1.986842, 1.236842],
                                    's2': [3.065789, 3.815789],
                                    's3': [2.526316, 2.526316],
                                    's4': [3.605263, 5.105263],
                                    's5': [3.065789, 3.815789],
                                    's6': [4.144737, 6.394737],
                                    's7': [3.605263, 5.105263]},
                                   index=['Y1', 'Y2']).T

        pdt.assert_frame_equal(res_predict, exp_predict)


if __name__ == "__main__":
    unittest.main()
