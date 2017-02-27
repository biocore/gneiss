import os
from gneiss.plot import heatmap
import pandas as pd
from skbio import TreeNode, DistanceMatrix
from scipy.cluster.hierarchy import ward
from gneiss.plot._dendrogram import SquareDendrogram
import numpy as np
import numpy.testing.utils as npt
import unittest


class HeatmapTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.table = pd.DataFrame(np.random.random((5, 5)))
        num_otus = 5  # otus
        x = np.random.rand(num_otus)
        dm = DistanceMatrix.from_iterable(x, lambda x, y: np.abs(x-y))
        lm = ward(dm.condensed_form())
        t = TreeNode.from_linkage_matrix(lm, np.arange(len(x)).astype(np.str))
        self.t = SquareDendrogram.from_tree(t)
        self.md = pd.Series(['a', 'a', 'a', 'b', 'b'])
        for i, n in enumerate(t.postorder()):
            if not n.is_tip():
                n.name = "y%d" % i
            n.length = np.random.rand()*3

        self.highlights = pd.DataFrame({'y8': ['#FF0000', '#00FF00'],
                                        'y6': ['#0000FF', '#F0000F']}).T

    def test_basic(self):
        fig = heatmap(self.table, self.t, self.md)

        # Test to see if the lineages of the tree are ok
        lines = list(fig.get_axes()[1].get_lines())
        pts = self.t.coords(width=20, height=self.table.shape[0])
        pts['y'] = pts['y'] - 0.5  # account for offset
        pts['x'] = pts['x'].astype(np.float)
        pts['y'] = pts['y'].astype(np.float)

        npt.assert_allclose(lines[0]._xy,
                            pts.loc[['y5', '3'], ['x', 'y']])
        npt.assert_allclose(lines[1]._xy,
                            pts.loc[['y6', '0'], ['x', 'y']].values)
        npt.assert_allclose(lines[2]._xy,
                            pts.loc[['y7', '1'], ['x', 'y']].values)
        npt.assert_allclose(lines[3]._xy,
                            pts.loc[['y8', '2'], ['x', 'y']].values)
        npt.assert_allclose(lines[4]._xy,
                            pts.loc[['y5', '4'], ['x', 'y']].values)
        npt.assert_allclose(lines[5]._xy,
                            pts.loc[['y6', 'y5'], ['x', 'y']].values)
        npt.assert_allclose(lines[6]._xy,
                            pts.loc[['y7', 'y6'], ['x', 'y']].values)
        npt.assert_allclose(lines[7]._xy,
                            pts.loc[['y8', 'y7'], ['x', 'y']].values)

        # Make sure that the metadata labels are set properly
        res = str(fig.get_axes()[0].get_xticklabels(minor=True)[0])
        self.assertEqual(res, "Text(0,0,'a')")

        res = str(fig.get_axes()[0].get_xticklabels(minor=True)[1])
        self.assertEqual(res, "Text(0,0,'b')")

        # make sure that xlims are set properly
        self.assertEqual(fig.get_axes()[0].get_xlim(),
                         (-0.5, 4.5))

        self.assertEqual(fig.get_axes()[1].get_xlim(),
                         (-1.0, 21.0))

        # make sure that ylims are set properly
        self.assertEqual(fig.get_axes()[0].get_ylim(),
                         (-0.5, 4.5))

        self.assertEqual(fig.get_axes()[1].get_ylim(),
                         (-0.5, 4.5))

    def test_basic_highlights(self):
        fig = heatmap(self.table, self.t, self.md, self.highlights)

        # Test to see if the lineages of the tree are ok
        lines = list(fig.get_axes()[1].get_lines())
        pts = self.t.coords(width=20, height=self.table.shape[0])
        pts['y'] = pts['y'] - 0.5  # account for offset
        pts['x'] = pts['x'].astype(np.float)
        pts['y'] = pts['y'].astype(np.float)

        npt.assert_allclose(lines[0]._xy,
                            pts.loc[['y5', '3'], ['x', 'y']].values)
        npt.assert_allclose(lines[1]._xy,
                            pts.loc[['y6', '0'], ['x', 'y']].values)
        npt.assert_allclose(lines[2]._xy,
                            pts.loc[['y7', '1'], ['x', 'y']].values)
        npt.assert_allclose(lines[3]._xy,
                            pts.loc[['y8', '2'], ['x', 'y']].values)
        npt.assert_allclose(lines[4]._xy,
                            pts.loc[['y5', '4'], ['x', 'y']].values)
        npt.assert_allclose(lines[5]._xy,
                            pts.loc[['y6', 'y5'], ['x', 'y']].values)
        npt.assert_allclose(lines[6]._xy,
                            pts.loc[['y7', 'y6'], ['x', 'y']].values)
        npt.assert_allclose(lines[7]._xy,
                            pts.loc[['y8', 'y7'], ['x', 'y']].values)

        # Make sure that the metadata labels are set properly
        res = str(fig.get_axes()[0].get_xticklabels(minor=True)[0])
        self.assertEqual(res, "Text(0,0,'a')")

        res = str(fig.get_axes()[0].get_xticklabels(minor=True)[1])
        self.assertEqual(res, "Text(0,0,'b')")

        # Make sure that the highlight labels are set properly
        res = str(fig.get_axes()[2].get_xticklabels()[0])
        self.assertEqual(res, "Text(0,0,'0')")

        res = str(fig.get_axes()[2].get_xticklabels()[0])
        self.assertEqual(res, "Text(0,0,'0')")

        # make sure that xlims are set properly
        self.assertEqual(fig.get_axes()[0].get_xlim(),
                         (-0.5, 4.5))

        self.assertEqual(fig.get_axes()[1].get_xlim(),
                         (-1.0, 21.0))

        self.assertEqual(fig.get_axes()[2].get_xlim(),
                         (0.0, 1.0))

        # make sure that ylims are set properly
        self.assertEqual(fig.get_axes()[0].get_ylim(),
                         (-0.5, 4.5))

        self.assertEqual(fig.get_axes()[1].get_ylim(),
                         (-0.5, 4.5))

        self.assertEqual(fig.get_axes()[1].get_ylim(),
                         (-0.5, 4.5))

if __name__ == "__main__":
    unittest.main()
