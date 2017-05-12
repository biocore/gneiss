# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------
import pickle
import abc


class Model(metaclass=abc.ABCMeta):

    def __init__(self, submodels, balances):
        """
        Abstract container for balance models.

        Parameters
        ----------
        submodels : list of statsmodels objects
            List of statsmodels result objects.
        balances : pd.DataFrame
            A table of balances where samples are rows and
            balances are columns. These balances were calculated
            using `tree`.
        """
        # TODO: Make sure that the tree and basis are strictly coupled
        # this will require the development of methods to convert
        # back and forth between these methods.
        self.submodels = submodels
        self.balances = balances
        self.results = []

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
