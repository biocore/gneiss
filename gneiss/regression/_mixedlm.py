# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------
from collections import OrderedDict
import pandas as pd
import statsmodels.formula.api as smf
from ._model import RegressionModel
from gneiss.util import _type_cast_to_float
from statsmodels.iolib.summary2 import Summary
from gneiss.balances import balance_basis
from skbio.stats.composition import ilr_inv


def mixedlm(formula, table, metadata, groups, **kwargs):
    """ Linear Mixed Effects Models applied to balances.

    Linear mixed effects (LME) models is a method for estimating
    parameters in a linear regression model with mixed effects.
    LME models are commonly used for repeated measures, where multiple
    samples are collected from a single source.  This implementation is
    focused on performing a multivariate response regression with mixed
    effects where the response is a matrix of balances (`table`), the
    covariates (`metadata`) are made up of external variables and the
    samples sources are specified by `groups`.

    T-statistics (`tvalues`) and p-values (`pvalues`) can be obtained to
    investigate to evaluate statistical significance for a covariate for a
    given balance.  Predictions on the resulting model can be made using
    (`predict`), and these results can be interpreted as either balances or
    proportions.

    Parameters
    ----------
    formula : str
        Formula representing the statistical equation to be evaluated.
        These strings are similar to how equations are handled in R.
        Note that the dependent variable in this string should not be
        specified, since this method will be run on each of the individual
        balances. See `patsy` [1]_ for more details.
    table : pd.DataFrame
        Contingency table where samples correspond to rows and
        balances correspond to columns.
    metadata: pd.DataFrame
        Metadata table that contains information about the samples contained
        in the `table` object.  Samples correspond to rows and covariates
        correspond to columns.
    groups : str
        Column name in `metadata` that specifies the groups.  These groups are
        often associated with individuals repeatedly sampled, typically
        longitudinally.
    **kwargs : dict
        Other arguments accepted into
        `statsmodels.regression.linear_model.MixedLM`

    Returns
    -------
    LMEModel
        Container object that holds information about the overall fit.
        This includes information about coefficients, pvalues and
        residuals from the resulting regression.

    References
    ----------
    .. [1] https://patsy.readthedocs.io/en/latest/

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from gneiss.regression import mixedlm

    Here, we will define a table of balances with features `Y1`, `Y2`
    across 12 samples.

    >>> table = pd.DataFrame({
    ...   'u1': [ 1.00000053,  6.09924644],
    ...   'u2': [ 0.99999843,  7.0000045 ],
    ...   'u3': [ 1.09999884,  8.08474053],
    ...   'x1': [ 1.09999758,  1.10000349],
    ...   'x2': [ 0.99999902,  2.00000027],
    ...   'x3': [ 1.09999862,  2.99998318],
    ...   'y1': [ 1.00000084,  2.10001257],
    ...   'y2': [ 0.9999991 ,  3.09998418],
    ...   'y3': [ 0.99999899,  3.9999742 ],
    ...   'z1': [ 1.10000124,  5.0001796 ],
    ...   'z2': [ 1.00000053,  6.09924644],
    ...   'z3': [ 1.10000173,  6.99693644]},
    ..     index=['Y1', 'Y2']).T

    Now we are going to define some of the external variables to
    test for in the model.  Here we will be testing a hypothetical
    longitudinal study across 3 time points, with 4 patients
    `x`, `y`, `z` and `u`, where `x` and `y` were given treatment `1`
    and `z` and `u` were given treatment `2`.

    >>> metadata = pd.DataFrame({
    ...         'patient': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
    ...         'treatment': [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
    ...         'time': [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]
    ...     }, index=['x1', 'x2', 'x3', 'y1', 'y2', 'y3',
    ...               'z1', 'z2', 'z3', 'u1', 'u2', 'u3'])

    Now we can run the linear mixed effects model on the balances.
    Underneath the hood, the proportions will be transformed into balances,
    so that the linear mixed effects models can be run directly on balances.
    Since each patient was sampled repeatedly, we'll specify them separately
    in the groups.  In the linear mixed effects model `time` and `treatment`
    will be simultaneously tested for with respect to the balances.

    >>> res = mixedlm('time + treatment', table, metadata,
    ...               groups='patient')

    See Also
    --------
    statsmodels.regression.linear_model.MixedLM
    ols

    """
    metadata = _type_cast_to_float(metadata.copy())
    data = pd.merge(table, metadata, left_index=True, right_index=True)
    if len(data) == 0:
        raise ValueError(("No more samples left.  Check to make sure that "
                          "the sample names between `metadata` and `table` "
                          "are consistent"))
    submodels = []
    for b in table.columns:
        # mixed effects code is obtained here:
        # http://stackoverflow.com/a/22439820/1167475
        stats_formula = '%s ~ %s' % (b, formula)
        mdf = smf.mixedlm(stats_formula, data=data,
                          groups=data[groups],
                          **kwargs)
        submodels.append(mdf)

    # ugly hack to get around the statsmodels object
    model = LMEModel(Y=table, Xs=None)
    model.submodels = submodels
    model.balances = table
    return model


class LMEModel(RegressionModel):
    """ Summary object for storing linear mixed effects results.

    A `LMEModel` object stores information about the
    individual balances used in the regression, the coefficients,
    residuals. This object can be used to perform predictions.
    In addition, summary statistics such as the coefficient
    of determination for the overall fit can be calculated.


    Attributes
    ----------
    Y : pd.DataFrame
        A table of balances where samples are rows and
        balances are columns. These balances were calculated
        using `tree`.
    Xs : pd.DataFrame
        Design matrix.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fit(self, **kwargs):
        """ Fit the model """
        # assumes that the underlying submodels have implemented `fit`.
        # TODO: Add regularized fit
        self.results = [s.fit(**kwargs) for s in self.submodels]

    def summary(self):
        """ Summarize the Linear Mixed Effects Regression Results.

        Parameters
        ----------
        ndim : int
            Number of dimensions to summarize for coefficients.
            If `ndim` is None, then all of the dimensions of the covariates
            will be printed.

        Returns
        -------
        str :
            This holds the summary of regression coefficients and fit
            information.
        """

        # number of observations
        self.nobs = self.balances.shape[0]
        self.model = None

        # Start filling in summary information
        smry = Summary()
        # Top results
        info = OrderedDict()
        info["No. Observations"] = self.balances.shape[0]
        info["Model:"] = "Simplicial MixedLM"
        smry.add_dict(info)
        smry.add_title("Simplicial Mixed Linear Model Results")
        # TODO: We need better model statistics
        return smry

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
            return coef.T

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
            A table of predicted values where rows are covariates,
            and the columns are balances. If `tree` is specified, then
            the columns are proportions.
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
        return pvals.T
