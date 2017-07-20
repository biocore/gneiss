# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------
from collections import OrderedDict
import numpy as np
import pandas as pd
from gneiss.regression._model import RegressionModel
from gneiss.util import _type_cast_to_float
from gneiss.balances import balance_basis
from skbio.stats.composition import ilr_inv

from statsmodels.iolib.summary2 import Summary
from statsmodels.sandbox.tools.cross_val import LeaveOneOut
from patsy import dmatrix
from scipy import stats


def ols(formula, table, metadata):
    """ Ordinary Least Squares applied to balances.

    An ordinary least squares (OLS) regression is a method for estimating
    parameters in a linear regression model.  OLS is a common statistical
    technique for fitting and testing the effects of covariates on a response.
    This implementation is focused on performing a multivariate response
    regression where the response is a matrix of balances (`table`) and the
    covariates (`metadata`) are made up of external variables.

    Global statistical tests indicating goodness of fit and contributions
    from covariates can be accessed from a coefficient of determination (`r2`),
    leave-one-variable-out cross validation (`lovo`), leave-one-out
    cross validation (`loo`) and k-fold cross validation (`kfold`).
    In addition residuals (`residuals`) can be accessed for diagnostic
    purposes.

    T-statistics (`tvalues`) and p-values (`pvalues`) can be obtained to
    investigate to evaluate statistical significance for a covariate for a
    given balance.  Predictions on the resulting model can be made using
    (`predict`), and these results can be interpreted as either balances or
    proportions.

    Parameters
    ----------
    formula : str
        Formula representing the statistical equation to be evaluated.
        These strings are similar to how equations are handled in R and
        statsmodels. Note that the dependent variable in this string should
        not be specified, since this method will be run on each of the
        individual balances. See `patsy` for more details.
    table : pd.DataFrame
        Contingency table where samples correspond to rows and
        balances correspond to columns.
    metadata: pd.DataFrame
        Metadata table that contains information about the samples contained
        in the `table` object.  Samples correspond to rows and covariates
        correspond to columns.

    Returns
    -------
    OLSModel
        Container object that holds information about the overall fit.
        This includes information about coefficients, pvalues, residuals
        and coefficient of determination from the resulting regression.

    Example
    -------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from skbio import TreeNode
    >>> from gneiss.regression import ols

    Here, we will define a table of balances as follows

    >>> np.random.seed(0)
    >>> n = 100
    >>> g1 = np.linspace(0, 15, n)
    >>> y1 = g1 + 5
    >>> y2 = -g1 - 2
    >>> Y = pd.DataFrame({'y1': y1, 'y2': y2})

    Once we have the balances defined, we will add some errors

    >>> e = np.random.normal(loc=1, scale=0.1, size=(n, 2))
    >>> Y = Y + e

    Now we will define the environment variables that we want to
    regress against the balances.

    >>> X = pd.DataFrame({'g1': g1})

    Once these variables are defined, a regression can be performed.
    These proportions will be converted to balances according to the
    tree specified.  And the regression formula is specified to run
    `temp` and `ph` against the proportions in a single model.

    >>> res = ols('g1', Y, X)
    >>> res.fit()

    From the summary results of the `ols` function, we can view the
    pvalues according to how well each individual balance fitted in the
    regression model.

    >>> res.pvalues
                          y1             y2
    Intercept  8.826379e-148   7.842085e-71
    g1         1.923597e-163  1.277152e-163

    We can also view the balance coefficients estimated in the regression
    model. These coefficients can also be viewed as proportions by passing
    `project=True` as an argument in `res.coefficients()`.

    >>> res.coefficients()
                     y1        y2
    Intercept  6.016459 -0.983476
    g1         0.997793 -1.000299

    The overall model fit can be obtained as follows

    >>> res.r2
    0.99945903186495066

    """

    # one-time creation of exogenous data matrix allows for faster run-time
    metadata = _type_cast_to_float(metadata.copy())
    x = dmatrix(formula, metadata, return_type='dataframe')
    ilr_table, x = table.align(x, join='inner', axis=0)
    return OLSModel(Y=ilr_table, Xs=x)


class OLSModel(RegressionModel):
    """ Summary object for storing ordinary least squares results.

    A `OLSModel` object stores information about the
    individual balances used in the regression, the coefficients,
    residuals. This object can be used to perform predictions.
    In addition, summary statistics such as the coefficient
    of determination for the overall fit can be calculated.


    Attributes
    ----------
    submodels : list of statsmodels objects
        List of statsmodels result objects.
    balances : pd.DataFrame
        A table of balances where samples are rows and
        balances are columns.  These balances were calculated
        using `tree`.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fit(self, **kwargs):
        """ Fit the ordinary least squares model.

        Here, the coefficients of the model are estimated.
        In addition, there are additional summary statistics
        that are being calculated, such as residuals, t-statistics,
        pvalues and coefficient of determination.


        Parameters
        ----------
        **kwargs : dict
           Keyword arguments used to tune the parameter estimation.
        """
        Y = self.response_matrix
        X = self.design_matrices

        n, p = X.shape
        inv = np.linalg.pinv(np.dot(X.T, X))
        cross = np.dot(inv, X.T)
        beta = np.dot(cross, Y)
        pX = np.dot(X, beta)
        resid = (Y - pX)
        sst = (Y - Y.mean(axis=0))
        sse = (resid**2).sum(axis=0)

        sst_balance = ((Y - Y.mean(axis=0))**2).sum(axis=0)

        sse_balance = (resid**2).sum(axis=0)
        ssr_balance = (sst_balance - sse_balance)

        df_resid = n - p + 1
        mse = sse / df_resid
        self._mse = mse
        # t tests
        cov = np.linalg.pinv(np.dot(X.T, X))
        bse = np.sqrt(np.outer(np.diag(cov), mse))
        tvalues = np.divide(beta, bse)
        pvals = stats.t.sf(np.abs(tvalues), df_resid)*2

        self._tvalues = pd.DataFrame(tvalues, index=X.columns,
                                     columns=Y.columns)
        self._pvalues = pd.DataFrame(pvals, index=X.columns,
                                     columns=Y.columns)
        self._beta = pd.DataFrame(beta, index=X.columns,
                                  columns=Y.columns)
        self._resid = pd.DataFrame(resid, index=Y.index,
                                   columns=Y.columns)
        self._fitted = True
        self._ess = ssr_balance
        self._r2 = 1 - ((resid**2).values.sum() / (sst**2).values.sum())

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
            A table of predicted values where columns are coefficients,
            and the rows are balances. If `tree` is specified, then
            the rows are proportions.

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

    @property
    def pvalues(self):
        """ Return pvalues from each of the coefficients in the fit. """
        return self._pvalues

    @property
    def tvalues(self):
        """ Return t-statistics from each of the coefficients in the fit. """
        return self._tvalues

    @property
    def r2(self):
        """ Coefficient of determination for overall fit"""
        return self._r2

    @property
    def mse(self):
        """ Mean Sum of squares Error"""
        return self._mse

    @property
    def ess(self):
        """ Explained Sum of squares"""
        return self._ess

    def summary(self, kfolds, lovo):
        """ Summarize the Ordinary Least Squares Regression Results.

        Parameters
        ----------
        kfold : pd.DataFrame
            Results from kfold cross-validation
        lovo : pd.DataFrame
            Results from leave-one-variable-out cross-validation.

        Returns
        -------
        str :
            This holds the summary of regression coefficients and fit
            information.
        """

        _r2 = self.r2

        self.params = self._beta

        # number of observations
        self.nobs = self.response_matrix.shape[0]
        self.model = None

        # Start filling in summary information
        smry = Summary()
        # Top results
        info = OrderedDict()
        info["No. Observations"] = self.nobs
        info["Model:"] = "OLS"
        info["Rsquared: "] = _r2

        # TODO: Investigate how to properly resize the tables
        smry.add_dict(info, ncols=1)
        smry.add_title("Simplicial Least Squares Results")
        smry.add_df(lovo, align='l')
        smry.add_df(kfolds, align='l')
        return smry

    def kfold(self, num_folds=10, **kwargs):
        """ K-fold cross-validation.

        Performs k-fold cross-validation by spliting the data
        into k partitions, building a model on k-1 partitions and
        evaluating the predictions on the remaining partition.
        This process is performed k times.

        Parameters
        ----------
        num_folds: int, optional
            The number of partitions used for the cross validation.
        **kwargs : dict
           Keyword arguments used to tune the parameter estimation.

        Returns
        -------
        pd.DataFrame
           model_mse : np.array, float
               The within model mean sum of squares error for each iteration of
               the cross validation.
           Rsquared : np.array, float
               The Rsquared of the model fitted on training data.
           pred_mse : np.array, float
               Prediction mean sum of squares error for each iteration of
               the cross validation.
        """
        # number of observations (i.e. samples)
        nobs = self.response_matrix.shape[0]
        s = nobs // num_folds
        folds = [np.arange(i*s, ((i*s)+s) % nobs) for i in range(num_folds)]
        results = pd.DataFrame(index=['fold_%d' % i for i in range(num_folds)],
                               columns=['model_mse', 'Rsquared', 'pred_mse'],
                               dtype=np.float64)

        for k in range(num_folds):
            test = folds[k]
            train = np.hstack(folds[:k] + folds[k+1:])

            res_i = OLSModel(self.response_matrix.iloc[train],
                             self.design_matrix.iloc[train])
            res_i.fit(**kwargs)

            # model error
            p = res_i.predict(X=self.design_matrix.iloc[train]).values
            r = self.response_matrix.iloc[train].values

            model_resid = ((p - r)**2)
            model_mse = np.mean(model_resid.sum(axis=0))

            results.loc['fold_%d' % k, 'model_mse'] = model_mse
            results.loc['fold_%d' % k, 'Rsquared'] = res_i.r2

            # prediction error
            p = res_i.predict(X=self.design_matrix.iloc[test]).values
            r = self.response_matrix.iloc[test].values

            pred_resid = ((p - r)**2)
            pred_mse = np.mean(pred_resid.sum(axis=0))

            results.loc['fold_%d' % k, 'pred_mse'] = pred_mse

        return results

    def loo(self, **kwargs):
        """ Leave one out cross-validation.

        Calculates summary statistics for each iteraction of
        leave one out cross-validation, specially `mse` on entire model
        and `pred_err` to measure prediction error.

        Parameters
        ----------
        **kwargs : dict
           Keyword arguments used to tune the parameter estimation.

        Returns
        -------
        pd.DataFrame
           model_mse : np.array, float
               Mean sum of squares error for each iteration of
               the cross validation.
           pred_mse : np.array, float
               Prediction mean sum of squares error for each iteration of
               the cross validation.

        See Also
        --------
        fit
        statsmodels.regression.linear_model.
        """
        # number of observations (i.e. samples)
        nobs = self.response_matrix.shape[0]
        cv_iter = LeaveOneOut(nobs)

        results = pd.DataFrame(index=self.response_matrix.index,
                               columns=['model_mse', 'pred_mse'],
                               dtype=np.float64)

        for i, (train, test) in enumerate(cv_iter):
            sample_id = self.response_matrix.index[i]

            res_i = OLSModel(self.response_matrix.iloc[train],
                             self.design_matrix.iloc[train])
            res_i.fit(**kwargs)

            # model error
            predicted = res_i.predict(X=self.design_matrix.iloc[train])
            r = self.response_matrix.iloc[train].values
            p = predicted.values
            model_resid = ((r - p)**2)
            model_mse = np.mean(model_resid.sum(axis=0))
            results.loc[sample_id, 'model_mse'] = model_mse

            # prediction error
            predicted = res_i.predict(X=self.design_matrix.iloc[test])
            r = self.response_matrix.iloc[test].values
            p = predicted.values
            pred_resid = ((r - p)**2)
            pred_mse = np.mean(pred_resid.sum(axis=0))
            results.loc[sample_id, 'pred_mse'] = pred_mse

        return results

    def lovo(self, **kwargs):
        """ Leave one variable out cross-validation.

        Calculates summary statistics for each iteraction of leave one variable
        out cross-validation, specially `r2` and `mse` on entire model.
        This technique is particularly useful for feature selection.

        Parameters
        ----------
        **kwargs : dict
           Keyword arguments used to tune the parameter estimation.

        Returns
        -------
        pd.DataFrame
           mse : np.array, float
               Mean sum of squares error for each iteration of
               the cross validation.
           Rsquared : np.array, float
               Coefficient of determination for each variable left out.
           R2diff : np.array, float
               Decrease in Rsquared for each variable left out.
        """
        cv_iter = LeaveOneOut(len(self.design_matrix.columns))
        results = pd.DataFrame(index=self.design_matrix.columns,
                               columns=['mse', 'Rsquared', 'R2diff'],
                               dtype=np.float64)
        for i, (inidx, outidx) in enumerate(cv_iter):

            feature_id = self.design_matrix.columns[i]

            res_i = OLSModel(Y=self.response_matrix,
                             Xs=self.design_matrix.iloc[:, inidx])
            res_i.fit(**kwargs)
            predicted = res_i.predict()
            r = self.response_matrix.values
            p = predicted.values

            model_resid = ((r - p)**2)
            model_mse = np.mean(model_resid.sum(axis=0))
            results.loc[feature_id, 'mse'] = model_mse
            results.loc[feature_id, 'Rsquared'] = res_i.r2
            results.loc[feature_id, 'R2diff'] = self.r2 - res_i.r2
        return results
