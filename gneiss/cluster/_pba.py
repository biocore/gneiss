# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------
import numpy as np
import pandas as pd
from gneiss.sort import mean_niche_estimator
from gneiss.util import match, rename_internal_nodes
from gneiss.composition._variance import variation_matrix

from skbio import TreeNode, DistanceMatrix
from scipy.spatial.distance import euclidean
from scipy.cluster.hierarchy import linkage


def correlation_linkage(X, method='ward'):
    r"""
    Hierarchical Clustering based on proportionality.

    The hierarchy is built based on the correlationity between
    any two pairs of features.  Specifically the correlation between
    two features :math:`x` and :math:`y` is measured by

    .. math::
        p(x, y) = var (\ln \frac{x}{y})

    If :math:`p(x, y)` is very small, then :math:`x` and :math:`y`
    are said to be highly correlation. A hierarchical clustering is
    then performed using this correlation as a distance metric.

    This can be useful for constructing principal balances [1]_.

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
        Tree for constructing principal balances.

    References
    ----------

    .. [1] Pawlowsky-Glahn V, Egozcue JJ, and Tolosana-Delgado R.
       Principal Balances (2011).

    Examples
    --------
    >>> import pandas as pd
    >>> from gneiss.cluster import correlation_linkage
    >>> table = pd.DataFrame([[1, 1, 0, 0, 0],
    ...                       [0, 1, 1, 0, 0],
    ...                       [0, 0, 1, 1, 0],
    ...                       [0, 0, 0, 1, 1]],
    ...                      columns=['s1', 's2', 's3', 's4', 's5'],
    ...                      index=['o1', 'o2', 'o3', 'o4']).T
    >>> tree = correlation_linkage(table+0.1)
    >>> print(tree.ascii_art())
                        /-o1
              /y1------|
             |          \-o2
    -y0------|
             |          /-o3
              \y2------|
                        \-o4
    """
    dm = variation_matrix(X)
    lm = linkage(dm.condensed_form(), method=method)
    t = TreeNode.from_linkage_matrix(lm, X.columns)
    t = rename_internal_nodes(t)
    return t


def rank_linkage(r, method='average'):
    r""" Hierchical Clustering on feature ranks.

    The hierarchy is built based on the rank values of the features given
    an input vector `r` of ranks. The distance between two features :math:`x`
    and :math:`y` can be defined as

    .. math::
       d(x, y) = (r(x) - r(y))^2

    Where :math:`r(x)` is the rank of the features.  Hierarchical clustering is
    then performed using :math:`d(x, y)` as the distance metric.

    This can be useful for constructing principal balances.

    Parameters
    ----------
    r : pd.Series
        Continuous vector representing some ordering of the features in X.
    method : str
        Clustering method.  (default='average')

    Returns
    -------
    skbio.TreeNode
        Tree for constructing principal balances.

    Examples
    --------
    >>> import pandas as pd
    >>> from gneiss.cluster import rank_linkage
    >>> ranks = pd.Series([1, 2, 4, 5],
    ...                   index=['o1', 'o2', 'o3', 'o4'])
    >>> tree = rank_linkage(ranks)
    >>> print(tree.ascii_art())
                        /-o1
              /y1------|
             |          \-o2
    -y0------|
             |          /-o3
              \y2------|
                        \-o4
    """
    dm = DistanceMatrix.from_iterable(r, euclidean)
    lm = linkage(dm.condensed_form(), method)
    t = TreeNode.from_linkage_matrix(lm, r.index)
    t = rename_internal_nodes(t)
    return t


def gradient_linkage(X, y, method='average'):
    r"""
    Hierarchical Clustering on known gradient.

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
    A hierarchical clustering is then performed using :math:`d(x, y)` as
    the distance metric.

    This can be useful for constructing principal balances.

    Parameters
    ----------
    X : pd.DataFrame
        Contingency table where the samples are rows and the features
        are columns.
    y : pd.Series
        Continuous vector representing some ordering of the samples in X.
    method : str
        Clustering method.  (default='average')

    Returns
    -------
    skbio.TreeNode
        Tree for constructing principal balances.

    See Also
    --------
    mean_niche_estimator

    Examples
    --------
    >>> import pandas as pd
    >>> from gneiss.cluster import gradient_linkage
    >>> table = pd.DataFrame([[1, 1, 0, 0, 0],
    ...                       [0, 1, 1, 0, 0],
    ...                       [0, 0, 1, 1, 0],
    ...                       [0, 0, 0, 1, 1]],
    ...                      columns=['s1', 's2', 's3', 's4', 's5'],
    ...                      index=['o1', 'o2', 'o3', 'o4']).T
    >>> gradient = pd.Series([1, 2, 3, 4, 5],
    ...                      index=['s1', 's2', 's3', 's4', 's5'])
    >>> tree = gradient_linkage(table, gradient)
    >>> print(tree.ascii_art())
                        /-o1
              /y1------|
             |          \-o2
    -y0------|
             |          /-o3
              \y2------|
                        \-o4
    """
    _X, _y = match(X, y)
    mean_X = mean_niche_estimator(_X, gradient=_y)
    t = rank_linkage(mean_X)
    return t


def random_linkage(n):
    """ Generates a tree with random topology.

    Parameters
    ----------
    n : int
        Number of nodes in the tree

    Returns
    -------
    skbio.TreeNode
    Random tree for constructing principal balances.

    Examples
    --------
    >>> from gneiss.cluster import random_linkage
    >>> tree = random_linkage(10)

    Notes
    -----
    The nodes will be labeled from 0 to n.
    """
    index = np.arange(n).astype(np.str)
    x = pd.Series(np.random.rand(n), index=index)
    t = rank_linkage(x)
    return t
