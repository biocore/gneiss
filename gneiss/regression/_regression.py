# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the GPLv3 License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------


import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from skbio.stats.composition import ilr
from gneiss.util import match, match_tips, rename_internal_nodes
from gneiss.regression._summary import RegressionResults
from gneiss.balances import balance_basis


def _intersect_of_table_metadata_tree(table, metadata, tree):
    """ Matches tips, features and samples between the table, metadata
    and tree.  This module returns the features and samples that are
    contained in all 3 objects.

    Parameters
    ----------
    table : pd.DataFrame
        Contingency table where samples correspond to rows and
        features correspond to columns.
    metadata: pd.DataFrame
        Metadata table that contains information about the samples contained
        in the `table` object.  Samples correspond to rows and covariates
        correspond to columns.
    tree : skbio.TreeNode
        Tree object where the leaves correspond to the columns contained in
        the table.

    Returns
    -------
    pd.DataFrame
        Subset of `table` with common row names as `metadata`
        and common columns as `tree.tips()`
    pd.DataFrame
        Subset of `metadata` with common row names as `table`
    skbio.TreeNode
        Subtree of `tree` with common tips as `table`
    """
    if np.any(table <= 0):
        raise ValueError('Cannot handle zeros or negative values in `table`. '
                         'Use pseudocounts or ``multiplicative_replacement``.'
                         )
    # check to see if there are overlapping nodes in tree and table
    overlap = {n.name for n in tree.traverse()} & set(table.columns)
    if len(overlap) == 0:
        raise ValueError('There are no internal nodes in `tree` after'
                         'intersection with `table`.')

    _table, _metadata = match(table, metadata)
    _table, _tree = match_tips(_table, tree)
    non_tips_no_name = [(n.name is None) for n in _tree.levelorder()
                        if not n.is_tip()]

    if any(non_tips_no_name):
        _tree = rename_internal_nodes(_tree)
    return _table, _metadata, _tree


def _to_balances(table, tree):
    """ Converts a table of abundances to balances given a tree.

    Parameters
    ----------
    table : pd.DataFrame
        Contingency table where samples correspond to rows and
        features correspond to columns.
    tree : skbio.TreeNode
        Tree object where the leaves correspond to the columns contained in
        the table.

    Returns
    -------
    pd.DataFrame
        Contingency table where samples correspond to rows and
        balances correspond to columns.
    np.array
        Orthonormal basis in the Aitchison simplex generated from `tree`.
    """
    non_tips = [n.name for n in tree.levelorder() if not n.is_tip()]
    basis, _ = balance_basis(tree)

    mat = ilr(table.values, basis=basis)
    ilr_table = pd.DataFrame(mat,
                             columns=non_tips,
                             index=table.index)
    return ilr_table, basis


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
        features correspond to columns.
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
    RegressionResults
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

    table, metadata, tree = _intersect_of_table_metadata_tree(table,
                                                              metadata,
                                                              tree)
    ilr_table, basis = _to_balances(table, tree)
    data = pd.merge(ilr_table, metadata, left_index=True, right_index=True)

    fits = []

    for b in ilr_table.columns:
        # mixed effects code is obtained here:
        # http://stackoverflow.com/a/22439820/1167475
        stats_formula = '%s ~ %s' % (b, formula)

        mdf = smf.ols(stats_formula, data=data, **kwargs).fit()
        fits.append(mdf)

    return RegressionResults(fits, basis=basis,
                             feature_names=table.columns,
                             balances=ilr_table,
                             tree=tree)


def mixedlm(formula, table, metadata, tree, groups, **kwargs):
    """ Linear Mixed Effects Models applied to balances.

    A linear mixed effects model is performed on nonzero relative abundance
    data given a list of covariates, or explanatory variables such as ph,
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
        balances. See `patsy` for more details.
    table : pd.DataFrame
        Contingency table where samples correspond to rows and
        features correspond to columns.
    metadata: pd.DataFrame
        Metadata table that contains information about the samples contained
        in the `table` object.  Samples correspond to rows and covariates
        correspond to columns.
    tree : skbio.TreeNode
        Tree object where the leaves correspond to the columns contained in
        the table.
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

    fits = []
    for b in ilr_table.columns:
        # mixed effects code is obtained here:
        # http://stackoverflow.com/a/22439820/1167475
        stats_formula = '%s ~ %s' % (b, formula)
        mdf = smf.mixedlm(stats_formula, data=data,
                          groups=data[groups],
                          **kwargs).fit()
        fits.append(mdf)

    return RegressionResults(fits, basis=basis,
                             feature_names=table.columns,
                             balances=ilr_table,
                             tree=tree)
