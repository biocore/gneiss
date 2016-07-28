#!/usr/bin/env python

# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the GPLv3 License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------
import pandas as pd


class RegressionResults():
    """
    Summary object for storing regression results.
    """
    def __init__(self, stat_results,
                 feature_names=None,
                 basis=None):
        """ Reorganizes statsmodels regression modules.

        Accepts a list of statsmodels RegressionResults objects
        and performs some addition summary statistics.

        Parameters
        ----------
        stat_results : list, sm.RegressionResults
            List of RegressionResults objects.
        feature_names : array_like, str, optional
            List of original names for features.
        basis : np.array, optional
            Orthonormal basis in the Aitchison simplex.
            If this is not specified, then `project` cannot
            be enabled in `coefficients` or `predict`.
        """
        self.feature_names = feature_names
        self.basis = basis
        self.results = stat_results

        # sum of squares error.  Also referred to as sum of squares residuals
        sse = 0
        # sum of squares regression. Also referred to as
        # explained sum of squares.
        ssr = 0
        # See `statsmodels.regression.linear_model.RegressionResults`
        # for more explanation on `ess` and `ssr`.

        # obtain pvalues
        self.pvalues = pd.DataFrame()
        for r in self.results:
            p = r.pvalues
            p.name = r.model.endog_names
            self.pvalues = self.pvalues.append(p)
            sse += r.ssr
            ssr += r.ess

        # calculate the overall coefficient of determination (i.e. R2)
        sst = sse + ssr
        self.r2 = 1 - sse / sst
