from gneiss.correlation import lovell_distance
from gneiss.sort import mean_niche_estimator
from gneiss.util import match

from skbio import TreeNode, DistanceMatrix
from scipy.spatial.distance import euclidean
from scipy.cluster.hierarchy import average


def hcpba(X):
    """
    Principal Balance Analysis using Hierarchical Clustering.

    Parameters
    ----------
    X : pd.DataFrame
        Contingency table where the samples are rows and the features
        are columns.

    Returns
    -------
    skbio.TreeNode
        Tree generated from principle balance analysis

    Refererences
    ------------

    """
    dm = lovell_distance(X)
    lm = average(dm.condensed_form())
    return TreeNode.from_linkage_matrix(lm, X.columns)


# TODO: Add in parameter for passing in different types of
# clustering methods
def supgma(X, y):
    """
    Supervised hierarical clustering using UPGMA.

    Parameters
    ----------
    X : pd.DataFrame
        Contingency table where the samples are rows and the features
        are columns.
    y : pd.Series
        Continuous vector representing some ordering of the features in X.

    Returns
    -------
    skbio.TreeNode
        Tree generated from principle balance analysis

    Refererences
    ------------

    """
    _X, _y = match(X, y)
    mean_X = mean_niche_estimator(_X, gradient=_y)
    dm = DistanceMatrix.from_iterable(mean_X, euclidean)
    lm = average(dm.condensed_form())
    return TreeNode.from_linkage_matrix(lm, X.columns)
