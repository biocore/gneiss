# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------
import pickle
import abc
import numpy as np


class Model(metaclass=abc.ABCMeta):

    def __init__(self, Y, Xs):
        """
        Abstract container for balance models.

        Parameters
        ----------
        Y : pd.DataFrame
            Response matrix.  This is the matrix being predicted.
            Also known as the dependent variable in univariate analysis.
        Xs : iterable of pd.DataFrame
            Design matrices.  Also known as the independent variable
            in univariate analysis. Note that this allows for multiple
            design matrices to be inputted to enable multiple data block
            analysis.
        """
        self.response_matrix = Y
        self.design_matrices = Xs

    @abc.abstractmethod
    def fit(self, **kwargs):
        pass

    @abc.abstractmethod
    def summary(self):
        """ Print summary results """
        pass

    @classmethod
    def read_pickle(self, filename):
        """ Reads Model object from pickle file.

        Parameters
        ----------
        filename : str or filehandle
            Input file to unpickle.

        Returns
        -------
        Model

        Notes
        -----
        Warning: Loading pickled data received from untrusted
        sources can be unsafe. See: https://wiki.python.org/moin/UsingPickle
        """

        if isinstance(filename, str):
            with open(filename, 'rb') as fh:
                res = pickle.load(fh)
        else:
            res = pickle.load(filename)

        return res

    def write_pickle(self, filename):
        """ Writes Model object to pickle file.

        Parameters
        ----------
        filename : str or filehandle
            Output file to store pickled object.
        """
        if isinstance(filename, str):
            with open(filename, 'wb') as fh:
                pickle.dump(self, fh)
        else:
            pickle.dump(self, filename)

    def percent_explained(self):
        """ Proportion explained by each principal balance."""
        # Using sum of squares error calculation (df=1)
        # instead of population variance (df=0).
        axis_vars = np.var(self.response_matrix, ddof=1, axis=0)
        return axis_vars / axis_vars.sum()
