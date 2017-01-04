import numpy as np
from skbio import DistanceMatrix


def variation_matrix(X):
    """ Calculate Aitchison variation matrix.

    Parameters
    ----------
    X : np.array
        Contingency table where the samples are rows and the features
        are columns.
    Returns
    -------
    np.array
        Total variation matrix
    """
    v = np.zeros((X.shape[1], X.shape[1]))
    for i in range(X.shape[1]):
        for j in range(i):
            v[i, j] = np.var(np.log(X.iloc[:, i]) - np.log(X.iloc[:, j]))
    return DistanceMatrix((v + v.T) / 2, ids=X.columns)
