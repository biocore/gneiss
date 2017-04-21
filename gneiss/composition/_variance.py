# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------
import numpy as np
from skbio import DistanceMatrix
from skbio.stats.composition import closure


def variation_matrix(X):
    """ Calculate Aitchison variation matrix.

    This calculates the Aitchison variation matrix.  Given a compositional
    matrix :math:`X`, and columns :math:`i` and :math:`j`, the :math:`ij` entry
    in the variation matrix of :math:`X` is given by

    .. math:
        V_{ij} = \frac{1}{2} var(\ln \frac{x_i}{x_j})

    Parameters
    ----------
    X : pd.DataFrame
        Contingency table where there are n rows corresponding to samples
        and p features corresponding to columns.

    Returns
    -------
    skbio.DistanceMatrix
        Total variation matrix of size n x n.

    References
    ----------
    .. [1] V. Pawlowsky-Glahn, J. J. Egozcue, R. Tolosana-Delgado (2015),
       Modeling and Analysis of Compositional Data, Wiley, Chichester, UK

    .. [2] J. J. Egozcue, V. Pawlowsky-Glahn (2004), Groups of Parts and
       Their Balances in Compositional Data Analysis, Mathematical Geology
    """
    v = np.zeros((X.shape[1], X.shape[1]))
    x = closure(X)
    for i in range(X.shape[1]):
        for j in range(i):
            v[i, j] = np.var(np.log(x[:, i]) - np.log(x[:, j]))
    # Making matrix symmetry since V(ln (x/y) ) = V(ln (y/x) )
    # Also dividing by 2, to ensure unit norm for balances.
    # See Eqn 4 in [2]
    return DistanceMatrix((v + v.T) / 2, ids=X.columns)
