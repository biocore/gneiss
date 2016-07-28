#!/usr/bin/env python

# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the GPLv3 License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------
import pandas as pd
import pandas.util.testing as pdt
import numpy as np
import statsmodels.formula.api as smf
import unittest
from gneiss._summary import RegressionResults


class TestRegressionResults(unittest.TestCase):

    def setUp(self):

        data = pd.DataFrame([[1, 3, 4, 5, 2, 3, 4],
                             list(range(1, 8)),
                             [1, 3, 2, 4, 3, 5, 4]],
                            columns=['s1', 's2', 's3', 's4', 's5', 's6', 's7'],
                            index=['Y1', 'Y2', 'X']).T
        model1 = smf.ols(formula="Y1 ~ X", data=data)
        model2 = smf.ols(formula="Y2 ~ X", data=data)
        self.results = [model1.fit(), model2.fit()]


    def test_check_projection(self):
        pass

    def test_r2(self):
        pass

    def test_regression_results_coefficient(self):
        exp_coef = pd.DataFrame({'Intercept' : [1.447368, -0.052632],
                                 'X' : [0.539474, 1.289474]},
                                index=['Y1', 'Y2'])
        res = RegressionResults(self.results)
        pdt.assert_frame_equal(res.coefficients(), exp_coef,
                               check_exact=False,
                               check_less_precise=True)

    def test_regression_results_coefficient_projection(self):
        pass

    def test_regression_results_coefficient_project_error(self):
        exp_coef = pd.DataFrame({'Intercept' : [1.447368, -0.052632],
                                 'X' : [0.539474, 1.289474]},
                                index=['Y1', 'Y2'])
        res = RegressionResults(self.results)
        pdt.assert_frame_equal(res.coefficients(), exp_coef,
                               check_exact=False,
                               check_less_precise=True)
        pass

    def test_regression_results_residuals(self):
        pass

    def test_regression_results_residuals_projection(self):
        pass

    def test_regression_results_predict(self):
        pass

    def test_regression_results_predict_projection(self):
        pass

    def test_regression_results_pvalues(self):
        # checks to see if pvalues are calculated correctly.
        res = RegressionResults(self.results)
        exp = pd.DataFrame({'Intercept':[0.307081, 0.972395],
                            'X': [0.211391, 0.029677]},
                            index=['Y1', 'Y2'])
        pdt.assert_frame_equal(res.pvalues, exp,
                               check_exact=False,
                               check_less_precise=True)


if __name__ == "__main__":
    unittest.main()
