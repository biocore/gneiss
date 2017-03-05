# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------
from gneiss.sort import mean_niche_estimator
from gneiss.util import match
from gneiss.stats.composition import variation_matrix

from skbio import TreeNode, DistanceMatrix
from scipy.spatial.distance import euclidean
from scipy.cluster.hierarchy import linkage


def proportional_linkage(X, method='ward'):
    """
    Principal Balance Analysis using Hierarchical Clustering
    based on proportionality.

    The hierarchy is built based on the proportionality between
    any two pairs of features.  Specifically the proportionality between
    two features :math:`x` and :math:`y` is measured by

    .. math::
        p(x, y) = var \ln \frac{x}{y}

    If :math:`p(x, y)` is very small, then :math:`x` and :math:`y`
    are said to be highly proportional. A hierarchical clustering is
    then performed using this proportionality as a distance metric.

    Parameters
    ----------
    X : pd.DataFrame
        Contingency table where the samples are rows and the features
        are columns.
    method : str
        Clustering method.  (default='ward')

    Returns
    -------
    skbio.TreeNode
        Tree generated from principal balance analysis.

    Refererences
    ------------
    .. [1] Pawlowsky-Glahn V, Egozcue JJ, and Tolosana-Delgado R.
       Principal Balances (2011).
    """
    dm = variation_matrix(X)
    lm = linkage(dm.condensed_form(), method=method)
    return TreeNode.from_linkage_matrix(lm, X.columns)


def gradient_linkage(X, y, method='average'):
    """
    Principal Balance Analysis using Hierarchical Clustering
    on known gradient.

    The hierarchy is built based on the values of the samples
    located along a gradient.  Given a feature :math:`x`, the mean gradient
    values that :math:`x` was observed in is calculated by

    .. math::
        f(g , x) =
         \sum\limits_{i=1}^N g_i \frac{x_i}{\sum\limits_{j=1}^N x_j}

    Where :math:`N` is the number of samples, :math:`x_i` is the proportion of
    feature :math:`x` in sample :math:`i`, :math:`g_i` is the gradient value
    at sample `i`.

    The distance between two features :math:`x` and :math:`y` can be defined as

    .. math::
        d(x, y) = (f(g, x) - f(g, y))^2

    If :math:`d(x, y)` is very small, then :math:`x` and :math:`y`
    are expected to live in very similar positions across the gradient.
    A hierarchical clustering is  then performed using :math:`d(x, y)` as
    the distance metric.

    Parameters
    ----------
    X : pd.DataFrame
        Contingency table where the samples are rows and the features
        are columns.
    y : pd.Series
        Continuous vector representing some ordering of the features in X.
    method : str
        Clustering method.  (default='average')

    Returns
    -------
    skbio.TreeNode
        Tree generated from principal balance analysis.

    See Also
    --------
    mean_niche_estimator
    """
    _X, _y = match(X, y)
    mean_X = mean_niche_estimator(_X, gradient=_y)
    dm = DistanceMatrix.from_iterable(mean_X, euclidean)
    lm = linkage(dm.condensed_form(), method)
    return TreeNode.from_linkage_matrix(lm, X.columns)
