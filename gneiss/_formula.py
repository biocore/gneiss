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
    non_tips = [n.name for n in tree.levelorder() if not n.is_tip()]
    basis, _ = balance_basis(tree)

    mat = ilr(table.values, basis=basis)
    ilr_table = pd.DataFrame(mat,
                             columns=non_tips,
                             index=table.index)
    return ilr_table, basis


def ols(formula, table, metadata, tree, **kwargs):
    """ Ordinary Least Squares applied to balances.

    A ordinary least square regression is applied to each balance.

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
    ilr_table, basis = _transform(_table, _tree)
    data = pd.merge(ilr_table, _metadata, left_index=True, right_index=True)

    fits = []
    for b in ilr_table.columns:
        # mixed effects code is obtained here:
        # http://stackoverflow.com/a/22439820/1167475
        stats_formula = '%s ~ %s' % (b, formula)

        mdf = smf.ols(stats_formula, data=data).fit()
        fits.append(mdf)
    return RegressionResults(fits, basis=basis,
                             feature_names=table.columns)


def glm(formula, table, metadata, tree, **kwargs):
    """ Generalized Linear Models applied to balances.

    Parameters
    ----------
    """
    pass


def mixedlm(formula, table, metadata, tree, **kwargs):
    """ Linear Mixed Models applied to balances.

    Parameters
    ----------
    """
    pass


def gee(formula, table, metadata, tree, **kwargs):
    """ Generalized Estimating Equations applied to balances.

    Parameters
    ----------
    """
    pass
