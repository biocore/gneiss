from gneiss.plot import heatmap
from gneiss.plot._heatmap import _sort_table

import pandas as pd
import pandas.util.testing as pdt
from skbio import TreeNode, DistanceMatrix
from scipy.cluster.hierarchy import ward
from gneiss.plot._dendrogram import SquareDendrogram
from gneiss.util import block_diagonal
from gneiss.cluster import rank_linkage
import numpy as np
import numpy.testing.utils as npt
import unittest


class HeatmapTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.table = pd.DataFrame(np.random.random((5, 5)),
                                  index=['0', '1', '2', '3', '4'],
                                  columns=['0', '1', '2', '3', '4'])

        num_otus = 5  # otus
        x = np.random.rand(num_otus)
        dm = DistanceMatrix.from_iterable(x, lambda x, y: np.abs(x-y))
        lm = ward(dm.condensed_form())
        t = TreeNode.from_linkage_matrix(lm, np.arange(len(x)).astype(np.str))
        self.t = SquareDendrogram.from_tree(t)
        self.md = pd.Series(['a', 'a', 'a', 'b', 'b'],
                            index=['0', '1', '2', '3', '4'])
        for i, n in enumerate(t.postorder()):
            if not n.is_tip():
                n.name = "y%d" % i
            n.length = np.random.rand()*3

        self.highlights = pd.DataFrame({'y8': ['#FF0000', '#00FF00'],
                                        'y6': ['#0000FF', '#F0000F']}).T

    def test_sort_table(self):
        table = pd.DataFrame(
            [[1, 1, 0, 0, 0],
             [0, 1, 1, 0, 0],
             [0, 0, 1, 1, 0],
             [0, 0, 0, 1, 1]],
            columns=['s1', 's2', 's3', 's4', 's5'],
            index=['o1', 'o2', 'o3', 'o4'])
        mdvar = pd.Series(['a', 'b', 'a', 'b', 'a'],
                          index=['s1', 's2', 's3', 's4', 's5'])
        res_table, res_mdvar = _sort_table(table, mdvar)
        pdt.assert_index_equal(pd.Index(['s1', 's3', 's5', 's2', 's4']),
                               res_mdvar.index)
        pdt.assert_index_equal(pd.Index(['s1', 's3', 's5', 's2', 's4']),
                               res_table.columns)

    def test_basic(self):
        fig = heatmap(self.table, self.t, self.md,
                      figsize=(5, self.table.shape[0]))

        # Test to see if the lineages of the tree are ok
        lines = list(fig.get_axes()[0].get_lines())

        exp_coords = np.array([[14.25, 0.5],
                               [14.25, 1.],
                               [14.25, 1.],
                               [20., 1.],
                               [9.5, 1.25],
                               [9.5, 2.],
                               [9.5, 2.],
                               [20., 2.],
                               [4.75, 2.125],
                               [4.75, 3.],
                               [4.75, 3.],
                               [20., 3.],
                               [0., 3.0625],
                               [0., 4.],
                               [0., 4.],
                               [20., 4.],
                               [14.25, 0.5],
                               [14.25, 0.],
                               [14.25, 0.],
                               [20., 0.],
                               [9.5, 1.25],
                               [9.5, 0.5],
                               [9.5, 0.5],
                               [14.25, 0.5],
                               [4.75, 2.125],
                               [4.75, 1.25],
                               [4.75, 1.25],
                               [9.5, 1.25],
                               [0., 3.0625],
                               [0., 2.125],
                               [0., 2.125],
                               [4.75, 2.125]])

        res = np.vstack([i._xy for i in lines])

        npt.assert_allclose(exp_coords, res)

        # Make sure that the metadata labels are set properly
        res = str(fig.get_axes()[1].get_xticklabels(minor=True)[0])
        self.assertEqual(res, "Text(0,0,'a')")

        res = str(fig.get_axes()[1].get_xticklabels(minor=True)[1])
        self.assertEqual(res, "Text(0,0,'b')")

        res = str(fig.get_axes()[1].get_xlabel())
        self.assertEqual(res, "None")

    def test_basic_line_width(self):
        fig = heatmap(self.table, self.t, self.md,
                      figsize=(5, self.table.shape[0]), linewidth=1)

        # Test to see if the lineages of the tree are ok
        lines = list(fig.get_axes()[1].get_lines())
        widths = [l.get_lw() for l in lines]
        np.allclose(widths, [1.0] * len(widths))

    def test_highlights(self):

        table = pd.DataFrame(block_diagonal(ncols=5, nrows=5, nblocks=2),
                             index=['0', '1', '2', '3', '4'],
                             columns=['0', '1', '2', '3', '4'])
        t = rank_linkage(pd.Series([1, 2, 3, 4, 5],
                                   index=['0', '1', '2', '3', '4']))
        t = SquareDendrogram.from_tree(t)
        md = pd.Series(['a', 'a', 'a', 'b', 'b'],
                       index=['0', '1', '2', '3', '4'])
        for i, n in enumerate(t.postorder()):
            if not n.is_tip():
                n.name = "y%d" % i
            n.length = np.random.rand()*3

        highlights = pd.DataFrame({'y8': ['#FF0000', '#00FF00'],
                                   'y7': ['#0000FF', '#F0000F']}).T

        fig = heatmap(table, t, md, highlights)

        # Test to see if the lineages of the tree are ok
        lines = list(fig.get_axes()[0].get_lines())

        pts = self.t.coords(width=20, height=self.table.shape[0])
        pts['y'] = pts['y'] - 0.5  # account for offset
        pts['x'] = pts['x'].astype(np.float)
        pts['y'] = pts['y'].astype(np.float)

        exp_coords = np.array([[6.33333333, 3.5],
                               [6.33333333, 4.],
                               [6.33333333, 4.],
                               [20., 4.],
                               [12.66666667, 0.5],
                               [12.66666667, 1.],
                               [12.66666667, 1.],
                               [20., 1.],
                               [6.33333333, 1.25],
                               [6.33333333, 2.],
                               [6.33333333, 2.],
                               [20., 2.],
                               [0., 2.375],
                               [0., 3.5],
                               [0., 3.5],
                               [6.33333333, 3.5],
                               [6.33333333, 3.5],
                               [6.33333333, 3.],
                               [6.33333333, 3.],
                               [20., 3.],
                               [12.66666667, 0.5],
                               [12.66666667, 0.],
                               [12.66666667, 0.],
                               [20., 0.],
                               [6.33333333, 1.25],
                               [6.33333333, 0.5],
                               [6.33333333, 0.5],
                               [12.66666667, 0.5],
                               [0., 2.375],
                               [0., 1.25],
                               [0., 1.25],
                               [6.33333333, 1.25]])

        res = np.vstack([i._xy for i in lines])

        npt.assert_allclose(exp_coords, res)

        # Make sure that the metadata labels are set properly
        res = str(fig.get_axes()[2].get_xticklabels(minor=True)[0])
        self.assertEqual(res, "Text(0,0,'a')")

        res = str(fig.get_axes()[2].get_xticklabels(minor=True)[1])
        self.assertEqual(res, "Text(0,0,'b')")

        # Make sure that the highlight labels are set properly
        res = str(fig.get_axes()[1].get_xticklabels()[0])
        self.assertEqual(res, "Text(0,0,'y7')")

        res = str(fig.get_axes()[1].get_xticklabels()[1])
        self.assertEqual(res, "Text(0,0,'y8')")

        # Test to see if the highlights are ok
        res = fig.get_axes()[2].get_position()._points
        exp = np.array([[0.24, 0.1],
                        [0.808, 0.9]])
        npt.assert_allclose(res, exp)


if __name__ == "__main__":
    unittest.main()
