import pickle
from skbio import TreeNode
import abc
from skbio.stats.composition import ilr_inv, closure
import numpy as np
import pandas as pd


class Model(metaclass=abc.ABCMeta):

    def __init__(self, submodels, basis, tree, balances):
        """
        Abstract container for balance models.

        Parameters
        ----------
        submodels : list of statsmodels objects
            List of statsmodels result objects.
        basis : pd.DataFrame
            Orthonormal basis in the Aitchison simplex.
            Row names correspond to the leafs of the tree
            and the column names correspond to the internal nodes
            in the tree. If this is not specified, then `project` cannot
            be enabled in `coefficients` or `predict`.
        tree : skbio.TreeNode
            Bifurcating tree that defines `basis`.
        balances : pd.DataFrame
            A table of balances where samples are rows and
            balances are columns. These balances were calculated
            using `tree`.
        """
        # TODO: Make sure that the tree and basis are strictly coupled
        # this will require the development of methods to convert
        # back and forth between these methods.
        self.submodels = submodels
        self.basis = basis

        # Make all nodes in the tree queriable
        self._nodes = {n.name: n for n in tree.levelorder()}

        self.tree = tree
        self.balances = balances
        self.results = []

    @abc.abstractmethod
    def fit(self, **kwargs):
        pass

    @abc.abstractmethod
    def summary(self):
        """ Print summary results """
        pass

    def split_balance(self, balance_name):
        """ Splits a balance into its log ratio components.

        Parameters
        ----------
        node : str
             Name of internal node in the tree to be retrieved for

        Returns
        -------
        pd.DataFrame
            Dataframe where the first column contains the numerator and the
            second column contains the deminator of the balance.
        """
        node = self.tree.find(balance_name)
        left = node.children[0]
        right = node.children[1]

        b = np.expand_dims(self.balances[balance_name].values, axis=1)
        # need to scale down by the number of children in subtrees
        # there is an erroneous factor 1/sqrt(2) from the ilr_inv below
        # so we'll need to remove that.
        p = closure(ilr_inv(b) * np.sqrt(2))
        return pd.DataFrame(p, columns=[left.name, right.name],
                            index=self.balances.index)

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
        res.tree = TreeNode.read([res._tree])

        return res

    def write_pickle(self, filename):
        """ Writes Model object to pickle file.

        Parameters
        ----------
        filename : str or filehandle
            Output file to store pickled object.
        """
        self._tree = str(self.tree)
        self.tree = None
        if isinstance(filename, str):
            with open(filename, 'wb') as fh:
                pickle.dump(self, fh)
        else:
            pickle.dump(self, filename)
