# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the GPLv3 License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------
import pandas as pd
from skbio.stats.composition import ilr_inv
from gneiss._model import Model


class RegressionModel(Model):
    def __init__(self, *args, **kwargs):
        """
        Summary object for storing regression results.

        A `RegressionResults` object stores information about the
        individual balances used in the regression, the coefficients,
        residuals. This object can be used to perform predictions.
        In addition, summary statistics such as the coefficient
        of determination for the overall fit can be calculated.

        Parameters
        ----------
        submodels : list of statsmodels objects
            List of statsmodels result objects.
        basis : pd.DataFrame
            Orthonormal basis in the Aitchison simplex.
            Row names correspond to the leafs of the tree
            and the column names correspond to the internal nodes
            in the tree. If this is not specified, then `project` cannot
            be enabled in `coefficients` or `predict`.
        tree : skbio.TreeNode
            Bifurcating tree that defines `basis`.
        balances : pd.DataFrame
            A table of balances where samples are rows and
            balances are columns.  These balances were calculated
            using `tree`.
        """
        super().__init__(*args, **kwargs)

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

        References
        ----------
        .. [1] Aitchison, J. "A concise guide to compositional data analysis,
           CDA work." Girona 24 (2003): 73-81.
        """
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
            return pd.DataFrame(c, index=self.basis.columns,
                                columns=coef.columns)
        else:
            return coef

    def residuals(self, project=False):
        """ Returns calculated residuals from fit.

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
            return pd.DataFrame(proj_resid, index=self.basis.columns,
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
                                columns=self.basis.columns,
                                index=prediction.columns)
        else:
            return prediction.T

    @property
    def pvalues(self):
        """ Return pvalues from each of the coefficients in the fit. """
        pvals = pd.DataFrame()
        for r in self.results:
            p = r.pvalues
            p.name = r.model.endog_names
            pvals = pvals.append(p)
        return pvals
