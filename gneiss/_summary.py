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

    def _check_projection(self, project):
        """
        Parameters
        ----------
        project : bool
           Specifies if a projection into the Aitchison simplex can be
           performed.

        Raises
        ------
        ValueError:
            Cannot perform projection into Aitchison simplex if `basis`
            is not specified.
        ValueError:
            Cannot perform projection into Aitchison simplex
            if `feature_names` is not specified.
        """
        if self.basis is None and project:
            raise ValueError("Cannot perform projection into Aitchison simplex"
                             "if `basis` is not specified.")

        if self.feature_names is None and project:
            raise ValueError("Cannot perform projection into Aitchison simplex"
                             "if `feature_names` is not specified.")

    def coefficients(self, project=False):
        """ Returns coefficients from fit.

        Parameters
        ----------
        project : bool, optional
            Specifies if coefficients should be projected back into
            the Aitchison simplex.  If false, the coefficients will be
            represented as balances  (default: False).

        Returns
        -------
        pd.DataFrame
            A table of values where columns are coefficients, and the index
            is either balances or proportions, depending on the value of
            `project`.

        Raises
        ------
        ValueError:
            Cannot perform projection into Aitchison simplex if `basis`
            is not specified.
        ValueError:
            Cannot perform projection into Aitchison simplex
            if `feature_names` is not specified.
        """
        self._check_projection(project)
        coef = pd.DataFrame()

        for r in self.results:
            c = r.params
            c.name = r.model.endog_names
            coef = coef.append(c)

        if project:
            # `check=True` due to type issue resolved here
            # https://github.com/biocore/scikit-bio/pull/1396
            c = ilr_inv(coef.values.T, basis=self.basis, check=False).T
            return pd.DataFrame(c, index=self.feature_names,
                                columns=coef.columns)
        else:
            return coef
