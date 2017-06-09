# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------
import abc
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
        self._beta = None
        self._resid = None
        self._fitted = False
        super().__init__(*args, **kwargs)
        # there is only one design matrix for regression
        self.design_matrix = self.design_matrices

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
            A table of coefficients where rows are covariates,
            and the columns are balances. If `tree` is specified, then
            the columns are proportions.
        """
        if not self._fitted:
            ValueError(('Model not fitted - coefficients not calculated.'
                        'See `fit()`'))
        coef = self._beta
        if tree is not None:
            basis, _ = balance_basis(tree)
            c = ilr_inv(coef.values, basis=basis)
            ids = [n.name for n in tree.tips()]
            return pd.DataFrame(c, columns=ids, index=coef.index)
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
            A table of residuals where rows are covariates,
            and the columns are balances. If `tree` is specified, then
            the columns are proportions.

        References
        ----------
        .. [1] Aitchison, J. "A concise guide to compositional data analysis,
           CDA work." Girona 24 (2003): 73-81.
        """
        if not self._fitted:
            ValueError(('Model not fitted - coefficients not calculated.'
                        'See `fit()`'))
        resid = self._resid
        if tree is not None:
            basis, _ = balance_basis(tree)
            proj_resid = ilr_inv(resid.values, basis=basis)
            ids = [n.name for n in tree.tips()]
            return pd.DataFrame(proj_resid,
                                columns=ids,
                                index=resid.index)
        else:
            return resid

    @abc.abstractmethod
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
            A table of predicted values where rows are covariates,
            and the columns are balances. If `tree` is specified, then
            the columns are proportions.

        """
        if not self._fitted:
            ValueError(('Model not fitted - coefficients not calculated.'
                        'See `fit()`'))
        if X is None:
            X = self.design_matrices

        prediction = X.dot(self._beta)
        if tree is not None:
            basis, _ = balance_basis(tree)
            proj_prediction = ilr_inv(prediction.values, basis=basis)
            ids = [n.name for n in tree.tips()]
            return pd.DataFrame(proj_prediction,
                                columns=ids,
                                index=prediction.index)
        else:
            return prediction
