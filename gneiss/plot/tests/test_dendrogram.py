# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------
import unittest
import numpy as np
import pandas as pd
from skbio import DistanceMatrix, TreeNode
from gneiss.plot._dendrogram import (Dendrogram, UnrootedDendrogram,
                                     SquareDendrogram)
from scipy.cluster.hierarchy import ward
import pandas.util.testing as pdt


class mock(Dendrogram):
    # mock dendrogram class to make sure that inheritance
    # is working as expected.
    def rescale(self, width, height):
        pass


class TestDendrogram(unittest.TestCase):

    def test_cache_ntips(self):
        dm = DistanceMatrix.from_iterable([0, 1, 2, 3],
                                          lambda x, y: np.abs(x-y))
        lm = ward(dm.condensed_form())
        ids = np.arange(4).astype(np.str)
        t = mock.from_linkage_matrix(lm, ids)

        t._cache_ntips()

        self.assertEquals(t.leafcount, 4)
        self.assertEquals(t.children[0].leafcount, 2)
        self.assertEquals(t.children[1].leafcount, 2)
        self.assertEquals(t.children[0].children[0].leafcount, 1)
        self.assertEquals(t.children[0].children[1].leafcount, 1)
        self.assertEquals(t.children[1].children[0].leafcount, 1)
        self.assertEquals(t.children[1].children[1].leafcount, 1)


class TestUnrootedDendrogram(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
        x = np.random.rand(10)
        dm = DistanceMatrix.from_iterable(x, lambda x, y: np.abs(x-y))
        lm = ward(dm.condensed_form())
        ids = np.arange(len(x)).astype(np.str)
        self.tree = TreeNode.from_linkage_matrix(lm, ids)

        # initialize tree with branch length and named internal nodes
        for i, n in enumerate(self.tree.postorder(include_self=True)):
            n.length = 1
            if not n.is_tip():
                n.name = "y%d" % i

    def test_from_tree(self):
        t = UnrootedDendrogram.from_tree(self.tree)
        self.assertEqual(t.__class__, UnrootedDendrogram)

    def test_coords(self):
        t = UnrootedDendrogram.from_tree(self.tree)

        exp = pd.DataFrame({'0': [404.097, 396.979, np.nan, np.nan, True],
                            '1': [464.724, 174.338, np.nan, np.nan, True],
                            '2': [487.5, 43.2804, np.nan, np.nan, True],
                            '3': [446.172, 359.095, np.nan, np.nan, True],
                            '4': [32.4704, 456.72, np.nan, np.nan, True],
                            '5': [438.468, 14.9717, np.nan, np.nan, True],
                            '6': [81.5024, 485.028, np.nan, np.nan, True],
                            '7': [54.5748, 34.9421, np.nan, np.nan, True],
                            '8': [12.5, 72.8265, np.nan, np.nan, True],
                            '9': [55.2464, 325.662, np.nan, np.nan, True],
                            'y10': [366.837, 313.291, '0', '3', False],
                            'y14': [419.421, 104.579, '2', '5', False],
                            'y15': [373.617, 183.914, '1', 'y14', False],
                            'y16': [305.539, 245.212, 'y10', 'y15', False],
                            'y17': [214.432, 254.788, 'y7', 'y16', False],
                            'y18': [153.134, 186.709, 'y2', 'y17', False],
                            'y2': [91.8354, 118.631, '7', '8', False],
                            'y6': [100.549, 395.421, '4', '6', False],
                            'y7': [146.353, 316.086, '9', 'y6', False]},
                           index=['x', 'y', 'child0', 'child1', 'is_tip']).T

        res = t.coords(500, 500)
        pdt.assert_frame_equal(exp, res)

    def test_rescale(self):
        t = UnrootedDendrogram.from_tree(self.tree)
        self.assertAlmostEqual(t.rescale(500, 500), 91.608680314971238,
                               places=5)

    def test_update_coordinates(self):
        t = UnrootedDendrogram.from_tree(self.tree)
        exp = pd.DataFrame([(-0.59847214410395644, -1.6334372886412185),
                            (-0.99749498660405445, -0.76155647142658189),
                            (1.0504174348855488, 0.34902579063315775),
                            (2.8507394969018511, 0.88932809650129752),
                            (3.3688089449017027, 0.082482736278627664),
                            (0.81247946938427551, -3.4080712447257464),
                            (-0.13677590240930079, -3.5433843164696093),
                            (-1.6101831260150372, -1.1190611577178871),
                            (-1.6176088321192579, 0.76057470265451865),
                            (-0.69694851846105044, 1.0284925540912822)])

        res = pd.DataFrame(t.update_coordinates(1, 0, 0, 2, 1))
        pdt.assert_frame_equal(res, exp, check_less_precise=True)


class TestSquareDendrogram(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
        self.table = pd.DataFrame(np.random.random((5, 5)))
        num_otus = 5  # otus
        x = np.random.rand(num_otus)
        dm = DistanceMatrix.from_iterable(x, lambda x, y: np.abs(x-y))
        lm = ward(dm.condensed_form())
        t = TreeNode.from_linkage_matrix(lm, np.arange(len(x)).astype(np.str))
        self.tree = SquareDendrogram.from_tree(t)

        for i, n in enumerate(t.postorder()):
            if not n.is_tip():
                n.name = "y%d" % i
            n.length = np.random.rand()*3

    def test_from_tree(self):
        t = SquareDendrogram.from_tree(self.tree)
        self.assertEqual(t.__class__, SquareDendrogram)

    def test_coords(self):
        # just test to make sure that the coordinates are calculated properly.
        t = SquareDendrogram.from_tree(self.tree)

        exp = pd.DataFrame({'0': [20, 2.5, np.nan, np.nan, True],
                            '1': [20, 3.5, np.nan, np.nan, True],
                            '2': [20, 4.5, np.nan, np.nan, True],
                            '3': [20, 1.5, np.nan, np.nan, True],
                            '4': [20, 0.5, np.nan, np.nan, True],
                            'y5': [14.25, 1, '3', '4', False],
                            'y6': [9.5, 1.75, '0', 'y5', False],
                            'y7': [4.75, 2.625, '1', 'y6', False],
                            'y8': [0, 3.5625, '2', 'y7', False]},
                           index=['x', 'y', 'child0', 'child1', 'is_tip']).T

        res = t.coords(width=20, height=self.table.shape[0])
        pdt.assert_frame_equal(exp, res)

    def test_rescale(self):
        t = SquareDendrogram.from_tree(self.tree)
        res = t.rescale(10, 10)
        self.assertEqual(res, 2.5)


if __name__ == "__main__":
    unittest.main()
