import pandas as pd
from skbio.stats.composition import ilr_inv, clr_inv
import pickle
from skbio import TreeNode
import abc


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
            balances are columns.  These balances were calculated
            using `tree`.
        """
        self.submodels = submodels
        self.basis = basis

        # Make all nodes in the tree queriable
        self._nodes = {n.name: n for n in tree.levelorder()}

        self._tree = str(tree)
        self.balances = balances
        self.results = []

    def fit(self, **kwargs):
        """ Fit the model """
        for s in self.submodels:
            # assumes that the underlying submodels have implemented `fit`.
            m = s.fit(**kwargs)
            self.results.append(m)

    @abc.abstractmethod
    def summary(self):
        """ Print summary results """
        pass

    @property
    def tree(self):
        """ Return the underlying tree.

        Returns
        -------
        skbio.TreeNode
        """
        return TreeNode.read([self._tree])

    def inode(self, tname):
        """ Return the underlying subtree to tname.

        Parameters
        ----------
        tname : str
            name of the skbio.TreeNode to return

        Returns
        -------
        skbio.TreeNode
        """
        return self._nodes[tname]

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
