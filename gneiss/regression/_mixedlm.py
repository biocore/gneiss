# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the GPLv3 License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------
from collections import OrderedDict
import pandas as pd
import statsmodels.formula.api as smf
from ._model import RegressionModel
from ._regression import (_intersect_of_table_metadata_tree,
                          _to_balances)
from decimal import Decimal
from statsmodels.iolib.summary2 import Summary


def mixedlm(formula, table, metadata, tree, groups, **kwargs):
    """ Linear Mixed Effects Models applied to balances.

    A linear mixed effects model is performed on nonzero relative abundance
    data given a list of covariates, or explanatory variables such as pH,
    treatment, etc to test for specific effects. The relative abundance data
    is transformed into balances using the ILR transformation, using a tree to
    specify the groupings of the features. The linear mixed effects model is
    applied to each balance separately. Only positive data will be accepted,
    so if there are zeros present, consider using a zero imputation method
    such as ``skbio.stats.composition.multiplicative_replacement`` or
    add a pseudocount.

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
        features correspond to columns.
    metadata: pd.DataFrame
        Metadata table that contains information about the samples contained
        in the `table` object.  Samples correspond to rows and covariates
        correspond to columns.
    tree : skbio.TreeNode
        Tree object that defines the partitions of the features. Each of the
        leaves correspond to the columns contained in the table.
    groups : str
        Column names in `metadata` that specifies the groups.  These groups are
        often associated with individuals repeatedly sampled, typically
        longitudinally.
    **kwargs : dict
        Other arguments accepted into
        `statsmodels.regression.linear_model.MixedLM`

    Returns
    -------
    RegressionResults
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
    >>> from skbio.stats.composition import ilr_inv
    >>> from skbio import TreeNode
    >>> from gneiss.regression import mixedlm

    Here, we will define a table of proportions with 3 features
    `a`, `b`, and `c` across 12 samples.

    >>> table = pd.DataFrame({
    ...         'u1':  [0.804248, 0.195526, 0.000226],
    ...         'u2':  [0.804369, 0.195556, 0.000075],
    ...         'u3':  [0.825711, 0.174271, 0.000019],
    ...         'x1':  [0.751606, 0.158631, 0.089763],
    ...         'x2':  [0.777794, 0.189095, 0.033111],
    ...         'x3':  [0.817855, 0.172613, 0.009532],
    ...         'y1':  [0.780774, 0.189819, 0.029406],
    ...         'y2':  [0.797332, 0.193845, 0.008824],
    ...         'y3':  [0.802058, 0.194994, 0.002948],
    ...         'z1':  [0.825041, 0.174129, 0.000830],
    ...         'z2':  [0.804248, 0.195526, 0.000226],
    ...         'z3':  [0.825667, 0.174261, 0.000072]}
    ...         index=['a', 'b', 'c']).T

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

    Finally, we need to define a bifurcating tree used to convert the
    proportions to balances.  If the internal nodes aren't labels,
    a default labeling will be applied (i.e. `y1`, `y2`, ...)

    >>> tree = TreeNode.read(['(c, (b,a)Y2)Y1;'])
    >>> print(tree.ascii_art())
              /-c
    -Y1------|
             |          /-b
              \Y2------|
                        \-a

    Now we can run the linear mixed effects model on the proportions.
    Underneath the hood, the proportions will be transformed into balances,
    so that the linear mixed effects models can be run directly on balances.
    Since each patient was sampled repeatedly, we'll specify them separately
    in the groups.  In the linear mixed effects model `time` and `treatment`
    will be simultaneously tested for with respect to the balances.

    >>> res = mixedlm('time + treatment', table, metadata, tree,
    ...               groups='patient')

    See Also
    --------
    statsmodels.regression.linear_model.MixedLM
    skbio.stats.composition.multiplicative_replacement
    ols

    """
    table, metadata, tree = _intersect_of_table_metadata_tree(table,
                                                              metadata,
                                                              tree)
    ilr_table, basis = _to_balances(table, tree)
    data = pd.merge(ilr_table, metadata, left_index=True, right_index=True)
    if len(data) == 0:
        raise ValueError(("No more samples left.  Check to make sure that "
                          "the sample names between `metadata` and `table` "
                          "are consistent"))
    submodels = []
    for b in ilr_table.columns:
        # mixed effects code is obtained here:
        # http://stackoverflow.com/a/22439820/1167475
        stats_formula = '%s ~ %s' % (b, formula)
        mdf = smf.mixedlm(stats_formula, data=data,
                          groups=data[groups],
                          **kwargs)
        submodels.append(mdf)

    return LMEModel(submodels, basis=basis,
                    balances=ilr_table, tree=tree)


class LMEModel(RegressionModel):
    def __init__(self, *args, **kwargs):
        """
        Summary object for storing linear mixed effects results.

        A `LMEModel` object stores information about the
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
            in the tree.
        tree : skbio.TreeNode
            Bifurcating tree that defines `basis`.
        balances : pd.DataFrame
            A table of balances where samples are rows and
            balances are columns. These balances were calculated
            using `tree`.
        """
        super().__init__(*args, **kwargs)

    def fit(self, **kwargs):
        """ Fit the model """
        for s in self.submodels:
            # assumes that the underlying submodels have implemented `fit`.
            # TODO: Add regularized fit
            m = s.fit(**kwargs)
            self.results.append(m)

    def summary(self, ndim=10):
        """ Summarize the Ordinary Least Squares Regression Results.

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

        # calculate the aitchison norm for all of the coefficients
        coefs = self.coefficients()
        if ndim:
            coefs = coefs.head(ndim)
        coefs.insert(0, '     ', ['slope']*coefs.shape[0])
        # We need a hierarchical index.  The outer index for each balance
        # and the inner index for each covariate
        if ndim:
            pvals = self.pvalues.head(ndim)
        # adding blank column just for the sake of display
        pvals.insert(0, '     ', ['pvalue']*pvals.shape[0])
        scores = pd.concat((coefs, pvals))
        scores = scores.sort_values(by='     ', ascending=False)
        scores = scores.sort_index()

        def _format(x):
            # format scores to be printable
            if x.dtypes == float:
                return ["%3.2E" % Decimal(k) for k in x]
            else:
                return x

        scores = scores.apply(_format)
        # TODO: Will want to add results for Aitchison norm
        # cnorms = pd.DataFrame({c: euclidean(0, coefs[c].values)
        #                        for c in coefs.columns}, index=['A-Norm']).T
        # cnorms = cnorms.apply(_format)

        self.params = coefs
        # TODO: Will want results from Hotelling t-test

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
        # TODO
        # smry.add_df(cnorms, align='r')
        smry.add_df(scores, align='r')

        return smry
