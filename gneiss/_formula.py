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


def _process(table, metadata, tree):
    """ Matches tips, features and samples between the table, metadata
    and tree.  This module returns the features and samples that are
    contained in all 3 objects.
    """
    _table, _metadata = match(table, metadata)
    _table, _tree = match_tips(table, tree)
    non_tips_no_name = [(n.name is None) for n in tree.levelorder()
                        if n.is_tip()]

    if any(non_tips_no_name):
        _tree = rename_internal_nodes(_tree)
    return _table, _metadata, _tree


def _transform(table, tree):
    """ Performs ILR transform on table"""
    basis, _ = balance_basis(tree)
    non_tips = [n.name for n in tree.levelorder() if not n.is_tip()]
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
    **kwargs : dict
        Other arguments accepted into `statsmodels.regression.linear_model.OLS`

    Returns
    -------
    RegressionResults
        Container object that holds information about the overall fit.

    See Also
    --------
    statsmodels.regression.linear_model.OLS
    """
    _table, _metadata, _tree = _process(table, metadata, tree)
    basis, ilr_table = _transform(_table, _tree)

    data = pd.merge(ilr_table, _metadata, left_index=True, right_index=True)

    fits = []
    for b in ilr_table.columns:
        # mixed effects code is obtained here:
        # http://stackoverflow.com/a/22439820/1167475
        stats_formula = '%s ~ %s' % (b, formula)

        mdf = smf.ols(stats_formula, data=data, **kwargs).fit()
        fits.append(mdf)
    return RegressionResults(fits, basis=basis,
                             feature_names=table.columns)


def glm(formula, table, metadata, tree, **kwargs):
    """ Generalized Linear Models applied to balances.

    Parameters
    ----------
    """
    pass


def mixedlm(formula, table, metadata, tree, groups, **kwargs):
    """ Linear Mixed Effects Models applied to balances.

    An Linear Mixed Effects model is applied to each balance.

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
        Variable in `metadata` that specifies the groups.  These groups are
        often associated with individuals repeatedly sampled, typically
        longitudinally.

    **kwargs : dict
        Other arguments accepted into `statsmodels.regression.linear_model.OLS`

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
    >>> from gneiss import mixedlm

    Here, we will define a table of proportions with 3 features
    `a`, `b`, and `c` across 12 samples.

    >>> table = pd.DataFrame({
    ...         'x1': ilr_inv(np.array([1.1, 1.1])),
    ...         'x2': ilr_inv(np.array([1., 2.])),
    ...         'x3': ilr_inv(np.array([1.1, 3.])),
    ...         'y1': ilr_inv(np.array([1., 2.1])),
    ...         'y2': ilr_inv(np.array([1., 3.1])),
    ...         'y3': ilr_inv(np.array([1., 4.])),
    ...         'z1': ilr_inv(np.array([1.1, 5.])),
    ...         'z2': ilr_inv(np.array([1., 6.1])),
    ...         'z3': ilr_inv(np.array([1.1, 7.])),
    ...         'u1': ilr_inv(np.array([1., 6.1])),
    ...         'u2': ilr_inv(np.array([1., 7.])),
    ...         'u3': ilr_inv(np.array([1.1, 8.1]))},
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
    Underneath the hood, the proportions will be transformed into balance,
    so that the linear mixed effects models can be run directly on balances.
    Since each patient was  sampled repeatedly, we'll specify them separately
    in the groups.  In the linear mixed effects  model `time` and `treatment`
    will be simultaneously tested for with respect to the balances.

    >>> res = mixedlm('time + treatment', table, metadata, tree, groups='patient')

    See Also
    --------
    statsmodels.regression.linear_model.MixedLM
    """
    _table, _metadata, _tree = _process(table, metadata, tree)
    ilr_table, basis = _transform(_table, _tree)
    data = pd.merge(ilr_table, _metadata, left_index=True, right_index=True)

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
                             feature_names=table.columns)


def gee(formula, table, metadata, tree, **kwargs):
    """ Generalized Estimating Equations applied to balances.

    Parameters
    ----------
    """
    pass
