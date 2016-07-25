import numpy as np
import pandas as pd
from functools import partial
from gneiss.util import match


def mean_niche_estimator(abundances, gradient):
    """ Estimates the mean niche of an organism.

    Calculates the mean niche or an organism along a gradient.
    This is done by calculating the expected value of an organism
    across the gradient.

    Parameters
    ----------
    abundances : pd.Series, np.float
        Vector of fraction abundances of an organism over
        a list of samples.
    gradient : pd.Series, np.float
        Vector of numerical gradient values.

    Returns
    -------
    np.float :
        The mean gradient that the organism lives in.
    """
    if len(abundances) != len(gradient):
        raise ValueError("Length of abundances (%d) doesn't match the length"
                         " of the gradient (%d)" % (len(abundances),
                                                    len(gradient)))
    if np.any(pd.isnull(gradient)):
        raise ValueError("`gradient` cannot have any missing values.")

    # normalizes the proportions of the organism across all of the
    # samples to add to 1.
    v = abundances / abundances.sum()
    return np.dot(gradient, v)


def nichesort(table, gradient, niche_estimator=None):
    """ Sort the table according to estimated niches.

    Sorts the table by samples along the gradient
    and otus by their estimated niche along the gradient.

    Parameters
    ----------
    table : pd.DataFrame
        Contingency table where samples are rows and
        features (i.e. OTUs) are columns.
    gradient : pd.Series
        Vector of numerical gradient values.
    niche_estimator : function
        A function that takes in two pandas series and returns an ordered
        object. The ability for the object to be ordered is critical, since
        this will allow the table to be sorted according to this sorting.

    Returns
    -------
    pd.DataFrame :
        Sorted table according to the gradient of the samples, and the niches
        of the organisms along that gradient.
    """

    if niche_estimator is None:
        niche_estimator = mean_niche_estimator

    niche_estimator = partial(niche_estimator,
                              gradient=gradient)

    _table, _gradient = match(table, gradient, intersect=True)
    norm_table = _table.apply(lambda x: x/x.sum(), axis=1)
    est_niche = norm_table.apply(niche_estimator, axis=0)

    _gradient = _gradient.sort_values()
    est_niche = est_niche.sort_values()

    _table = _table.reindex(index=_gradient.index)
    _table = _table.reindex(columns=est_niche.index)
    return _table
