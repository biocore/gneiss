# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the GPLv3 License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------
import pandas as pd
import statsmodels.formula.api as smf
from skbio.stats.composition import ilr
from gneiss.util import match, match_tips, rename_internal_nodes
from gneiss._summary import RegressionResults
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
    _table, _metadata = match(table, metadata)
    _table, _tree = match_tips(_table, tree)
    non_tips_no_name = [(n.name is None) for n in _tree.levelorder()
                        if not n.is_tip()]
    if len(non_tips_no_name) == 0:
        raise ValueError('There are no internal nodes in `tree` after'
                         'intersection with `table`.')

    if len(_table.index) == 0:
        raise ValueError('There are no internal nodes in `table` after '
                         'intersection with `metadata`.')

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

    An ordinary least square regression is applied to each balance.

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
    >>> from gneiss import ols
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

        mdf = smf.ols(stats_formula, data=data).fit()
        fits.append(mdf)
    return RegressionResults(fits, basis=basis,
                             feature_names=table.columns)
