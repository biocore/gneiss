# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the GPLv3 License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------
import pandas as pd
from skbio.stats.composition import ilr_inv, clr_inv


class RegressionResults():
    """
    Summary object for storing regression results.
    """
    def __init__(self,
                 stat_results,
                 feature_names=None,
                 basis=None,
                 balances=None,
                 tree=None):
        """ Reorganizes statsmodels regression results modules.

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
        # basis is now handled differently
        # https://github.com/biocore/scikit-bio/pull/1396
        self.basis = clr_inv(basis)
        self.results = stat_results
        self.tree = tree
        self.balances = balances

        # obtain pvalues
        self.pvalues = pd.DataFrame()
        for r in self.results:
            p = r.pvalues
            p.name = r.model.endog_names
            self.pvalues = self.pvalues.append(p)

    @property
    def r2(self):
        """ Calculates the coefficients of determination """
        # Reason why we wanted to move this out was because not
        # all types of statsmodels regressions have this property.

        # See `statsmodels.regression.linear_model.RegressionResults`
        # for more explanation on `ess` and `ssr`.
        # sum of squares regression. Also referred to as
        # explained sum of squares.
        ssr = sum([r.ess for r in self.results])
        # sum of squares error.  Also referred to as sum of squares residuals
        sse = sum([r.ssr for r in self.results])
        # calculate the overall coefficient of determination (i.e. R2)
        sst = sse + ssr
        return 1 - sse / sst

    def _check_projection(self, project):
        """ Checks to make sure that the `ilr_inv` can be performed.

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
                             " if `basis` is not specified.")

        if self.feature_names is None and project:
            raise ValueError("Cannot perform projection into Aitchison simplex"
                             " if `feature_names` is not specified.")

    def coefficients(self, project=False):
        """ Returns coefficients from fit.

        Parameters
        ----------
        project : bool, optional
            Specifies if coefficients should be projected back into
            the Aitchison simplex [1]_.  If false, the coefficients will be
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

        References
        ----------
        .. [1] Aitchison, J. "A concise guide to compositional data analysis,
           CDA work." Girona 24 (2003): 73-81.
        """
        self._check_projection(project)
        coef = pd.DataFrame()

        for r in self.results:
            c = r.params
            c.name = r.model.endog_names
            coef = coef.append(c)

        if project:
            # `check=False`, due to a problem with error handling
            # addressed here https://github.com/biocore/scikit-bio/pull/1396
            # This will need to be fixed here:
            # https://github.com/biocore/gneiss/issues/34
            c = ilr_inv(coef.values.T, basis=self.basis, check=False).T
            return pd.DataFrame(c, index=self.feature_names,
                                columns=coef.columns)
        else:
            return coef

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
            the Aitchison simplex [1]_.  If false, the coefficients will be
            represented as balances  (default: False).

        Returns
        -------
        pd.DataFrame
            A table of values where rows are samples, and the columns
            are either balances or proportions, depending on the value of
            `project`.

        References
        ----------
        .. [1] Aitchison, J. "A concise guide to compositional data analysis,
           CDA work." Girona 24 (2003): 73-81.
        """
        self._check_projection(project)

        resid = pd.DataFrame()

        for r in self.results:
            err = r.resid
            err.name = r.model.endog_names
            resid = resid.append(err)

        if project:
            # `check=False`, due to a problem with error handling
            # addressed here https://github.com/biocore/scikit-bio/pull/1396
            # This will need to be fixed here:
            # https://github.com/biocore/gneiss/issues/34
            proj_resid = ilr_inv(resid.values.T, basis=self.basis,
                                 check=False).T
            return pd.DataFrame(proj_resid, index=self.feature_names,
                                columns=resid.columns).T
        else:
            return resid.T

    def predict(self, X=None, project=False, **kwargs):
        """ Performs a prediction based on model.

        Parameters
        ----------
        X : pd.DataFrame, optional
            Input table of covariates, where columns are covariates, and
            rows are samples.  If not specified, then the fitted values
            calculated from training the model will be returned.
        project : bool, optional
            Specifies if coefficients should be projected back into
            the Aitchison simplex [1]_.  If false, the coefficients will be
            represented as balances  (default: False).
        **kwargs : dict
            Other arguments to be passed into the model prediction.

        Returns
        -------
        pd.DataFrame
            A table of values where rows are coefficients, and the columns
            are either balances or proportions, depending on the value of
            `project`.

        References
        ----------
        .. [1] Aitchison, J. "A concise guide to compositional data analysis,
           CDA work." Girona 24 (2003): 73-81.
        """
        self._check_projection(project)

        prediction = pd.DataFrame()
        for m in self.results:
            # check if X is none.
            p = pd.Series(m.predict(X, **kwargs))
            p.name = m.model.endog_names
            if X is not None:
                p.index = X.index
            else:
                p.index = m.fittedvalues.index
            prediction = prediction.append(p)

        if project:
            # `check=False`, due to a problem with error handling
            # addressed here https://github.com/biocore/scikit-bio/pull/1396
            # This will need to be fixed here:
            # https://github.com/biocore/gneiss/issues/34
            proj_prediction = ilr_inv(prediction.values.T, basis=self.basis,
                                      check=False)
            return pd.DataFrame(proj_prediction,
                                columns=self.feature_names,
                                index=prediction.columns)
        return prediction.T
