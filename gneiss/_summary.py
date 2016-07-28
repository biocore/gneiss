#!/usr/bin/env python

# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the GPLv3 License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------
import pandas as pd
from skbio.stats.composition import ilr_inv


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
        feature_names : array_like, str
            List of original names for features.
        basis : np.array, optional
            Orthonormal basis in the Aitchison simplex.
            If this is not specified, then `project` cannot
            be enabled in `coefficients` or `predict`.
        """
        self.feature_names = feature_names
        self.basis = basis
        self.results = stat_results

        # obtain pvalues
        pvals = pd.DataFrame()
        for i in range(len(self.results)):
            p = self.results[i].pvalues
            p.name = self.results[i].model.endog_names
            pvals = pvals.append(p)
        self.pvalues = pvals

        # calculate the overall R2 value
        sse = sum([r.ssr for r in self.results])
        ssr = sum([r.ess for r in self.results])
        sst = sse + ssr
        self.r2 = 1 - sse / sst
