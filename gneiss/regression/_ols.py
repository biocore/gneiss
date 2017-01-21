# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the GPLv3 License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------
from decimal import Decimal
from collections import OrderedDict

import pandas as pd
from scipy.spatial.distance import euclidean
from skbio.stats.composition import ilr_inv
from gneiss.regression._model import RegressionModel
from ._regression import (_intersect_of_table_metadata_tree,
                          _to_balances)
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import Summary


# TODO: Register as qiime 2 method
def ols(formula, table, metadata, tree, **kwargs):
    """ Ordinary Least Squares applied to balances.

    A ordinary least square regression is performed on nonzero relative
    abundance data given a list of covariates, or explanatory variables
    such as ph, treatment, etc to test for specific effects. The relative
    abundance data is transformed into balances using the ILR transformation,
    using a tree to specify the groupings of the features. The regression
    is then performed on each balance separately. Only positive data will
    be accepted, so if there are zeros present, consider using a zero
    imputation method such as ``multiplicative_replacement`` or add a
    pseudocount.

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
        features correspond to columns.  The features could either
        correspond proportions or balances.
    metadata: pd.DataFrame
        Metadata table that contains information about the samples contained
        in the `table` object.  Samples correspond to rows and covariates
        correspond to columns.
    tree : skbio.TreeNode
        Tree object where the leaves correspond to the columns contained in
        the table.
    **kwargs : dict
        Other arguments accepted into `statsmodels.regression.linear_model.OLS`

    Returns
    -------
    OLSModel
        Container object that holds information about the overall fit.

    Example
    -------
    >>> from gneiss.regression import ols
    >>> from skbio import TreeNode
    >>> import pandas as pd

    Here, we will define a table of proportions with 3 features
    `a`, `b`, and `c` across 5 samples.

    >>> proportions = pd.DataFrame(
    ...     [[0.720463, 0.175157, 0.104380],
    ...      [0.777794, 0.189095, 0.033111],
    ...      [0.796416, 0.193622, 0.009962],
    ...      [0.802058, 0.194994, 0.002948],
    ...      [0.803731, 0.195401, 0.000868]],
    ...     columns=['a', 'b', 'c'],
    ...     index=['s1', 's2', 's3', 's4', 's5'])

    Now we will define the environment variables that we want to
    regress against the proportions.

    >>> env_vars = pd.DataFrame({
    ...     'temp': [20, 20, 20, 20, 21],
    ...     'ph': [1, 2, 3, 4, 5]},
    ...     index=['s1', 's2', 's3', 's4', 's5'])

    Finally, we need to define a bifurcating tree used to convert the
    proportions to balances.  If the internal nodes aren't labels,
    a default labeling will be applied (i.e. `y1`, `y2`, ...)

    >>> tree = TreeNode.read(['(c, (b,a)Y2)Y1;'])

    Once these 3 variables are defined, a regression can be performed.
    These proportions will be converted to balances according to the
    tree specified.  And the regression formula is specified to run
    `temp` and `ph` against the proportions in a single model.

    >>> res = ols('temp + ph', proportions, env_vars, tree)

    From the summary results of the `ols` function, we can view the
    pvalues according to how well each individual balance fitted in the
    regression model.

    >>> res.pvalues
           Intercept            ph      temp
    Y1  2.479592e-01  1.990984e-11  0.243161
    Y2  6.089193e-10  5.052733e-01  0.279805

    We can also view the balance coefficients estimated in the regression
    model. These coefficients can also be viewed as proportions by passing
    `project=True` as an argument in `res.coefficients()`.

    >>> res.coefficients()
        Intercept            ph      temp
    Y1  -0.000499  9.999911e-01  0.000026
    Y2   1.000035  2.865312e-07 -0.000002

    The balance residuals from the model can be viewed as follows.  Again,
    these residuals can be viewed as proportions by passing `project=True`
    into `res.residuals()`

    >>> res.residuals()
                  Y1            Y2
    s1 -4.121647e-06 -2.998793e-07
    s2  6.226749e-07 -1.602904e-09
    s3  1.111959e-05  9.028437e-07
    s4 -7.620619e-06 -6.013615e-07
    s5 -1.332268e-14 -2.375877e-14

    The predicted balances can be obtained as follows.  Note that the predicted
    proportions can also be obtained by passing `project=True` into
    `res.predict()`

    >>> res.predict()
              Y1        Y2
    s1  1.000009  0.999999
    s2  2.000000  0.999999
    s3  2.999991  0.999999
    s4  3.999982  1.000000
    s5  4.999999  0.999998

    The overall model fit can be obtained as follows

    >>> res.r2
    0.99999999997996369

    See Also
    --------
    statsmodels.regression.linear_model.OLS
    skbio.stats.composition.multiplicative_replacement
    """
    # TODO: clean up
    table, metadata, tree = _intersect_of_table_metadata_tree(table,
                                                              metadata,
                                                              tree)
    ilr_table, basis = _to_balances(table, tree)
    data = pd.merge(ilr_table, metadata, left_index=True, right_index=True)

    submodels = []

    for b in ilr_table.columns:
        # mixed effects code is obtained here:
        # http://stackoverflow.com/a/22439820/1167475
        stats_formula = '%s ~ %s' % (b, formula)

        mdf = smf.ols(stats_formula, data=data, **kwargs)
        submodels.append(mdf)

    return OLSModel(submodels, basis=basis,
                    balances=ilr_table,
                    tree=tree)


class OLSModel(RegressionModel):
    def __init__(self, *args, **kwargs):
        """
        Summary object for storing ordinary least squares results.

        A `OLSModel` object stores information about the
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

    def fit(self, **kwargs):
        """ Fit the model """
        for s in self.submodels:
            # assumes that the underlying submodels have implemented `fit`.
            m = s.fit(**kwargs)
            self.results.append(m)

    def summary(self, title=None, yname=None, xname=None, head=None):
        """ Summarize the Ordinary Least Squares Regression Results.

        Parameters
        ----------
        title : string, optional
            Title for the top table. If not None, then this replaces the
            default title
        yname : string, optional
            Default is `y`
        xname : list of strings, optional
            Default is `var_##` for ## in p the number of regressors
        head : int
            Number of dimensions to summarize for coefficients.
            If not specified, then all of the dimensions of the covariates
            will be printed.

        Returns
        -------
        str :
            This holds the summary of regression coefficients and fit
            information.
        """

        _c = self.coefficients()
        coefs = _c.copy()
        coefs.insert(0, '  ', ['slope']*coefs.shape[0])
        # We need a hierarchical index.  The outer index for each balance
        # and the inner index for each covariate
        pvals = self.pvalues
        pvals.insert(0, '  ', ['pvalue']*coefs.shape[0])
        scores = pd.concat((coefs, pvals))
        # adding blank column just for the sake of display
        scores = scores.sort_values(by='  ', ascending=False)
        scores = scores.sort_index()

        def _format(x):
            # format scores to be printable
            if x.dtypes == float:
                return ["%3.2E" % Decimal(k) for k in x]
            else:
                return x

        scores = scores.apply(_format)
        # TODO: Add sort measure of effect size for slopes.
        # Not sure if euclidean norm is the most appropriate.
        # See https://github.com/biocore/gneiss/issues/27
        # cnorms = pd.DataFrame({c: euclidean(0, _c[c].values)
        #                        for c in _c.columns}, index=['A-Norm']).T
        # cnorms = cnorms.apply(_format)
        # TODO: Will want results from Hotelling t-test
        _r2 = self.r2

        self.params = _c

        # number of observations
        self.nobs = self.balances.shape[0]
        self.model = None

        # Start filling in summary information
        smry = Summary()
        # Top results
        info = OrderedDict()
        info["No. Observations"] = self.balances.shape[0]
        info["Model:"] = "Simplical Ordinary Least Squares"
        info["Rsquared: "] = _r2

        # TODO: Investigate how to properly resize the tables
        smry.add_dict(info)
        smry.add_title("Simplical Ordinary Linear Regression Results")
        smry.add_df(scores, align='l')

        return smry

    @property
    def r2(self):
        """ Coefficient of determination for overall fit"""
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
