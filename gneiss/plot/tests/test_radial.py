import unittest
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import ward

from skbio import TreeNode, DistanceMatrix
from gneiss.plot._radial import radialplot
from gneiss.plot._dendrogram import UnrootedDendrogram


class TestRadial(unittest.TestCase):
    def setUp(self):

        self.coords = pd.DataFrame(
            [['487.5', '347.769', 'NaN', 'NaN', 'True'],
             ['12.5', '483.28', 'NaN', 'NaN', 'True'],
             ['324.897', '16.7199', 'NaN', 'NaN', 'True'],
             ['338.261', '271.728', '0', '2', 'False'],
             ['193.169', '365.952', '1', 'y3', 'False']],
            columns=['x', 'y', 'child0', 'child1', 'is_tip'],
            index=['0', '1', '2', 'y3', 'y4'])

    def test_basic_plot(self):
        self.maxDiff = None
        exp_edges = {'dest_node': ['0', '1', '2', 'y3'],
                     'edge_color': ['#00FF00', '#00FF00',
                                    '#00FF00', '#FF0000'],
                     'edge_width': [2, 2, 2, 2],
                     'src_node': ['y3', 'y4', 'y3', 'y4'],
                     'x0': [338.2612593838583,
                            193.1688862557773,
                            338.2612593838583,
                            193.1688862557773],
                     'x1': [487.5, 12.499999999999972,
                            324.89684138234867, 338.2612593838583],
                     'y0': [271.7282256126416,
                            365.95231443706376,
                            271.7282256126416,
                            365.95231443706376],
                     'y1': [347.7691620070637,
                            483.2800610261029,
                            16.719938973897143,
                            271.7282256126416]}

        exp_nodes = {'child0': [np.nan, np.nan, np.nan, '0', '1'],
                     'child1': [np.nan, np.nan, np.nan, '2', 'y3'],
                     'color': ['#1C9099', '#1C9099', '#1C9099',
                               '#FF999F', '#FF999F'],
                     'hover_var': [None, None, None, None, None],
                     'is_tip': [True, True, True, False, False],
                     'node_size': [10, 10, 10, 10, 10],
                     'x': [487.5,
                           12.499999999999972,
                           324.89684138234867,
                           338.26125938385832,
                           193.16888625577729],
                     'y': [347.7691620070637,
                           483.28006102610289,
                           16.719938973897143,
                           271.72822561264161,
                           365.95231443706376]}
        np.random.seed(0)
        num_otus = 3  # otus
        x = np.random.rand(num_otus)
        dm = DistanceMatrix.from_iterable(x, lambda x, y: np.abs(x-y))
        lm = ward(dm.condensed_form())
        t = TreeNode.from_linkage_matrix(lm, np.arange(len(x)).astype(np.str))
        t = UnrootedDendrogram.from_tree(t)
        # incorporate colors in tree
        for i, n in enumerate(t.postorder(include_self=True)):
            if not n.is_tip():
                n.name = "y%d" % i
                n.color = '#FF999F'
                n.edge_color = '#FF0000'
                n.node_size = 10
            else:
                n.color = '#1C9099'
                n.edge_color = '#00FF00'
                n.node_size = 10
            n.length = np.random.rand()*3
            n.edge_width = 2
        p = radialplot(t, node_color='color', edge_color='edge_color',
                       node_size='node_size', edge_width='edge_width')

        for e in exp_edges.keys():
            self.assertListEqual(
                list(p.renderers[0].data_source.data[e]),
                exp_edges[e])

        for e in exp_nodes.keys():
            self.assertListEqual(
                list(p.renderers[1].data_source.data[e]),
                exp_nodes[e])

        self.assertTrue(isinstance(t, TreeNode))


if __name__ == "__main__":
    unittest.main()
