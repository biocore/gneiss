# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------
import pandas as pd
from skbio.stats.composition import ilr_inv
from gneiss._model import Model
from gneiss.balances import balance_basis


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
        balances : pd.DataFrame
            A table of balances where samples are rows and
            balances are columns.  These balances were calculated
            using `tree`.
        """
        super().__init__(*args, **kwargs)

    def coefficients(self, tree=None):
        """ Returns coefficients from fit.

        Parameters
        ----------
        tree : skbio.TreeNode, optional
            The tree used to perform the ilr transformation.  If this
            is specified, then the prediction will be represented as
            proportions. Otherwise, if this is not specified, the prediction
            will be represented as balances. (default: None).

        Returns
        -------
        pd.DataFrame
            A table of values where rows are coefficients, and the columns
            are proportions, if `tree` is specified.
        """
        coef = pd.DataFrame()

        for r in self.results:
            c = r.params
            c.name = r.model.endog_names
            coef = coef.append(c)

        if tree is not None:
            basis, _ = balance_basis(tree)
            c = ilr_inv(coef.values.T, basis=basis).T

            return pd.DataFrame(c, index=[n.name for n in tree.tips()],
                                columns=coef.columns)
        else:
            return coef

    def residuals(self, tree=None):
        """ Returns calculated residuals from fit.

        Parameters
        ----------
        X : pd.DataFrame, optional
            Input table of covariates.  If not specified, then the
            fitted values calculated from training the model will be
            returned.
        tree : skbio.TreeNode, optional
            The tree used to perform the ilr transformation.  If this
            is specified, then the prediction will be represented
            as proportions. Otherwise, if this is not specified,
            the prediction will be represented as balances. (default: None).

        Returns
        -------
        pd.DataFrame
            A table of values where rows are coefficients, and the columns
            are proportions, if `tree` is specified.

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

        if tree is not None:
            basis, _ = balance_basis(tree)
            proj_resid = ilr_inv(resid.values.T, basis=basis).T
            return pd.DataFrame(proj_resid,
                                index=[n.name for n in tree.tips()],
                                columns=resid.columns).T
        else:
            return resid.T

    def predict(self, X=None, tree=None, **kwargs):
        """ Performs a prediction based on model.

        Parameters
        ----------
        X : pd.DataFrame, optional
            Input table of covariates, where columns are covariates, and
            rows are samples.  If not specified, then the fitted values
            calculated from training the model will be returned.
        tree : skbio.TreeNode, optional
            The tree used to perform the ilr transformation.  If this
            is specified, then the prediction will be represented
            as proportions. Otherwise, if this is not specified,
            the prediction will be represented as balances. (default: None).
        **kwargs : dict
            Other arguments to be passed into the model prediction.

        Returns
        -------
        pd.DataFrame
            A table of values where rows are coefficients, and the columns
            are proportions, if `tree` is specified.

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

        if tree is not None:
            basis, _ = balance_basis(tree)
            proj_prediction = ilr_inv(prediction.values.T, basis=basis)
            return pd.DataFrame(proj_prediction,
                                columns=[n.name for n in tree.tips()],
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
