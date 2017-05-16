# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------
import unittest
import os
import shutil

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import ward

from skbio import TreeNode, DistanceMatrix
from gneiss.plot._plot import dendrogram_heatmap
from qiime2 import MetadataCategory


class TestHeatmap(unittest.TestCase):

    def setUp(self):
        self.results = "results"
        os.mkdir(self.results)

    def tearDown(self):
        shutil.rmtree(self.results)

    def test_visualization(self):
        np.random.seed(0)
        num_otus = 500  # otus
        table = pd.DataFrame(np.random.random((num_otus, 5)),
                             index=np.arange(num_otus).astype(np.str)).T

        x = np.random.rand(num_otus)
        dm = DistanceMatrix.from_iterable(x, lambda x, y: np.abs(x-y))
        lm = ward(dm.condensed_form())
        t = TreeNode.from_linkage_matrix(lm, np.arange(len(x)).astype(np.str))

        for i, n in enumerate(t.postorder()):
            if not n.is_tip():
                n.name = "y%d" % i
            n.length = np.random.rand()*3

        md = MetadataCategory(
            pd.Series(['a', 'a', 'a', 'b', 'b']))

        dendrogram_heatmap(self.results, table, t, md)

        index_fp = os.path.join(self.results, 'index.html')
        self.assertTrue(os.path.exists(index_fp))

        with open(index_fp, 'r') as fh:
            html = fh.read()
            self.assertIn('<h1>Dendrogram heatmap</h1>',
                          html)

    def test_visualization_small(self):
        # tests the scenario where ndim > number of tips
        np.random.seed(0)
        num_otus = 11  # otus
        table = pd.DataFrame(np.random.random((num_otus, 5)),
                             index=np.arange(num_otus).astype(np.str)).T

        x = np.random.rand(num_otus)
        dm = DistanceMatrix.from_iterable(x, lambda x, y: np.abs(x-y))
        lm = ward(dm.condensed_form())
        t = TreeNode.from_linkage_matrix(lm, np.arange(len(x)).astype(np.str))

        for i, n in enumerate(t.postorder()):
            if not n.is_tip():
                n.name = "y%d" % i
            n.length = np.random.rand()*3

        md = MetadataCategory(
            pd.Series(['a', 'a', 'a', 'b', 'b']))

        dendrogram_heatmap(self.results, table, t, md)

        index_fp = os.path.join(self.results, 'index.html')
        self.assertTrue(os.path.exists(index_fp))

        with open(index_fp, 'r') as fh:
            html = fh.read()
            self.assertIn('<h1>Dendrogram heatmap</h1>',
                          html)


if __name__ == "__main__":
    unittest.main()
