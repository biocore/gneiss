"""
Correlation functions (:mod:`gneiss.correlation`)
===================================

.. currentmodule:: gneiss.correlation

This module contains correlation functions to calculate dissimilarities
between features.


Functions
---------

.. autosummary::
   :toctree: generated/

       lovell_distance
"""
# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the GPLv3 License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------
import numpy as np
import pandas as pd
from skbio.stats.composition import clr
from skbio.stats.distance import DissimilarityMatrix, DistanceMatrix


def lovell_distance(X):
    r"""
    Calculates proportional goodness of fit for all pairwise comparisons.

    The lovell proportionality metric is a compositional proportionality
    measure. Rather than attempting to infer how correlated different features
    are, this metric will attempt to infer how proportional different features
    are.

    The original lovell proportionality metric is denoted as follows

    .. math::
       \phi(\ln x, \ln y) = \frac{var(\ln (\frac{x}{y})}{var(\ln x)}

    The smaller :math:`\phi(\ln x, \ln y)` is, the more proportional
    the :math:`x` :math:`y` are.

    Since the proportionality metric is an assymmetric metric,
    this metric can be made symmetric by taking the average.  For example

    .. math:::
        d(x, y) = \frac{1}{2}(\phi(x,y) + \phi(y, x))

    Parameters
    ----------
    X : array_like.
        Contingency table where the samples are rows and the features
        are columns.

    Returns
    -------
    skbio.DistanceMatrix
        This is a matrix of averaged lovell proportionality coefficients.

    Note
    ----
    This method assumes that there are no zeroes present in the data.
    """

    def _proportionality_metric(_x, _y):
        return np.var(_x - _y) / np.var(_x)
    cX = clr(X)
    dm = DissimilarityMatrix.from_iterable(cX.T, _proportionality_metric)
    if isinstance(X, pd.DataFrame):
        dm.ids = X.columns

    dm = DistanceMatrix((1/2) * (dm.data + dm.data.T), ids=dm.ids)
    return dm
