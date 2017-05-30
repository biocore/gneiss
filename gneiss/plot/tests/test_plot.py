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
import pandas.util.testing as pdt
from scipy.cluster.hierarchy import ward

from skbio import TreeNode, DistanceMatrix
from gneiss.plot._plot import dendrogram_heatmap, balance_taxonomy
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


class TestBalanceTaxonomy(unittest.TestCase):

    def setUp(self):
        self.results = "results"
        os.mkdir(self.results)
        self.balances = pd.DataFrame(
            {'a': [-2, -1, 0, 1, 2],
             'b': [-2, 0, 0, 0, 0]},
            index=['a1', 'a2', 'a3', 'a4', 'a5']
        )
        self.tree = TreeNode.read([r'((k, q)d, ((x, y)a, z)b)c;'])
        self.taxonomy = pd.DataFrame(
            [['foo;barf;a;b;c;d;e', 1],
             ['foo;bark;f;g;h;i;j', 1],
             ['foo;bark;f;g;h;w;j', 1],
             ['nom;tu;k;l;m;n;o', 0.9],
             ['nom;tu;k;l;m;t;o', 0.9]],
            columns=['Taxon', 'Confidence'],
            index=['x', 'y', 'z', 'k', 'q'])

        self.balances = pd.DataFrame(
            [[1, 2, 3, 4, 5, 6, 7],
             [-3.1, -2.9, -3, 3, 2.9, 3.2, 3.1],
             [1, 1, 1, 1, 1, 1, 1],
             [3, 2, 1, 0, -1, -2, -3]],
            index=['d', 'a', 'b', 'c'],
            columns=['s1', 's2', 's3', 's4', 's5', 's6', 's7']
        ).T

        self.categorical = MetadataCategory(
            pd.Series(['a', 'a', 'a', 'b', 'b', 'b', 'b'],
                      index=['s1', 's2', 's3', 's4', 's5', 's6', 's7'],
                      name='categorical'))
        self.continuous = MetadataCategory(
            pd.Series(np.arange(7),
                      index=['s1', 's2', 's3', 's4', 's5', 's6', 's7'],
                      name='continuous'))

    def tearDown(self):
        shutil.rmtree(self.results)
        pass

    def test_balance_taxonomy(self):
        index_fp = os.path.join(self.results, 'index.html')
        balance_taxonomy(self.results, self.balances, self.tree,
                         self.taxonomy, balance_name='c')
        self.assertTrue(os.path.exists(index_fp))
        # test to make sure that the numerator file is there
        num_fp = os.path.join(self.results, 'numerator.csv')
        self.assertTrue(os.path.exists(num_fp))
        # test to make sure that the denominator file is there
        denom_fp = os.path.join(self.results, 'denominator.csv')
        self.assertTrue(os.path.exists(denom_fp))

        with open(index_fp, 'r') as fh:
            html = fh.read()
            self.assertIn('<h1>Balance Taxonomy</h1>', html)
            self.assertIn('Numerator taxa', html)
            self.assertIn('Denominator taxa', html)

        # extract csv files and test for contents
        exp = pd.DataFrame(
            [['foo', 'barf', 'a', 'b', 'c', 'd', 'e'],
             ['foo', 'bark', 'f', 'g', 'h', 'i', 'j'],
             ['foo', 'bark', 'f', 'g', 'h', 'w', 'j']],
            columns=['1', '2', '3', '4', '5', '6', '7'],
            index=['x', 'y', 'z'])
        res = pd.read_csv(num_fp, index_col=0)
        pdt.assert_frame_equal(exp, res.sort_index())

        exp = pd.DataFrame([['nom', 'tu', 'k', 'l', 'm', 't', 'o'],
                            ['nom', 'tu', 'k', 'l', 'm', 'n', 'o']],
                           columns=['1', '2', '3', '4', '5', '6', '7'],
                           index=['q', 'k']).sort_index()
        res = pd.read_csv(denom_fp, index_col=0)
        pdt.assert_frame_equal(exp, res.sort_index())

    def test_balance_taxonomy_tips(self):
        index_fp = os.path.join(self.results, 'index.html')
        balance_taxonomy(self.results, self.balances, self.tree,
                         self.taxonomy, balance_name='a')
        self.assertTrue(os.path.exists(index_fp))
        # test to make sure that the numerator file is there
        num_fp = os.path.join(self.results, 'numerator.csv')
        self.assertTrue(os.path.exists(num_fp))
        # test to make sure that the denominator file is there
        denom_fp = os.path.join(self.results, 'denominator.csv')
        self.assertTrue(os.path.exists(denom_fp))

        exp = pd.DataFrame(['foo', 'bark', 'f', 'g', 'h', 'i', 'j'],
                           index=['1', '2', '3', '4',
                                  '5', '6', '7'],
                           columns=['y']).T
        res = pd.read_csv(num_fp, index_col=0)
        pdt.assert_frame_equal(exp, res)

        res = pd.read_csv(denom_fp, index_col=0)
        exp = pd.DataFrame(['foo', 'barf', 'a', 'b', 'c', 'd', 'e'],
                           index=['1', '2', '3', '4', '5', '6', '7'],
                           columns=['x']).T
        pdt.assert_frame_equal(exp, res)

    def test_balance_taxonomy_categorical(self):
        balance_taxonomy(self.results, self.balances, self.tree,
                         self.taxonomy, balance_name='a',
                         metadata=self.categorical)

    def test_balance_taxonomy_continuous(self):
        balance_taxonomy(self.results, self.balances, self.tree,
                         self.taxonomy, balance_name='a',
                         metadata=self.continuous)

    def test_balance_taxonomy_genus(self):
        index_fp = os.path.join(self.results, 'index.html')
        balance_taxonomy(self.results, self.balances, self.tree,
                         self.taxonomy, balance_name='c',
                         taxa_level='6')
        self.assertTrue(os.path.exists(index_fp))
        # test to make sure that the numerator file is there
        num_fp = os.path.join(self.results, 'numerator.csv')
        self.assertTrue(os.path.exists(num_fp))
        # test to make sure that the denominator file is there
        denom_fp = os.path.join(self.results, 'denominator.csv')
        self.assertTrue(os.path.exists(denom_fp))


if __name__ == "__main__":
    unittest.main()
