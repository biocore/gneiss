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
        feature_names : array_list, str
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

    def _check_projection(self, project):
        """
        Parameters
        ----------
        project : bool
           Specifies if a projection into the Aitchison simplex can be performed.

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
        """
        self._check_projection(project)
        coef = pd.DataFrame()

        for i in range(len(self.results)):
            c = self.results[i].params
            c.name = self.results[i].model.endog_names
            coef = coef.append(c)

        if project:
            # check=True due to type issue resolved here
            # https://github.com/biocore/scikit-bio/pull/1396
            c = ilr_inv(coef.values.T, basis=self.basis, check=False).T
            c = pd.DataFrame(c, index=self.feature_names,
                             columns=coef.columns)
            return c
        else:
            return coef

    def predict(self, X, project=False):
        """ Performs a prediction based on model.

        Parameters
        ----------
        X : pd.DataFrame, optional
            Input table of covariates.  If not specified, then the
            fitted values calculated from training the model will be
            returned.
        project : bool, optional
            Specifies if coefficients should be projected back into
            the Aitchison simplex.  If false, the coefficients will be
            represented as balances  (default: False).

        Returns
        -------
        pd.DataFrame
            A table of values where rows are coefficients, and the columns
            are either balances or proportions, depending on the value of
            `project`.
        """
        self._check_projection(project)

        pass

    def residuals(self, project=False):
        """ Returns calculated residuals.

        Parameters
        ----------
        X : pd.DataFrame, optional
            Input table of covariates.  If not specified, then the
            fitted values calculated from training the model will be
            returned.
        project : bool, optional
            Specifies if coefficients should be projected back into
            the Aitchison simplex.  If false, the coefficients will be
            represented as balances  (default: False).

        Returns
        -------
        pd.DataFrame
            A table of values where rows are coefficients, and the columns
            are either balances or proportions, depending on the value of
            `project`.
        """
        self._check_projection(project)

        resid = pd.DataFrame()

        for i in range(len(self.results)):
            err = self.results[i].resid
            err.name = self.results[i].model.endog_names
            resid = resid.append(err)

        if project:
            # check=True due to type issue resolved here
            # https://github.com/biocore/scikit-bio/pull/1396
            proj_resid = ilr_inv(resid.values.T, basis=self.basis,
                                 check=False).T
            proj_resid = pd.DataFrame(proj_resid, index=self.feature_names,
                                      columns=resid.columns)
            return proj_resid
        else:
            return resid

    def summary(self):
        """ Returns a summary of the RegressionResults.

        Returns
        -------
        pd.DataFrame :
            A table of coefficients, pvalues and statistics

        """



class DifferentialAbundanceResults():
    """
    Summary object for storing differential abundance results.
    """
    def __init__(tvalues, fvalues, pvalues):
        """ Reorganizes statsmodels regression modules

        Accepts a list of statsmodels AnovaResults objects
        and performs some addition summary statistics.
        """


