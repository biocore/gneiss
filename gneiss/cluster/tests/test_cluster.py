# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------
import unittest
import qiime2
import pandas as pd
from skbio.util import get_data_path
from skbio import TreeNode


class TestClusteringPlugin(unittest.TestCase):

    def test_proportional_artifact(self):
        from qiime2.plugins.gneiss.methods import proportional_clustering
        table_f = get_data_path("test_composition.biom.qza")
        in_table = qiime2.Artifact.load(table_f)

        res = proportional_clustering(in_table)
        res_clust = res.clustering._view(TreeNode)
        exp_str = ('((F4:0.228723591874,(F5:0.074748541601,'
                   '(F1:0.00010428164962,F2:0.00010428164962):'
                   '0.0746442599513):0.153975050273):0.70266138894,'
                   '(F3:0.266841737789,F6:0.266841737789):0.664543243026);\n')
        self.assertEqual(exp_str, str(res_clust))

    def test_gradient_artifact(self):
        from qiime2.plugins.gneiss.methods import gradient_clustering
        table_f = get_data_path("test_gradient.biom.qza")
        metadata_f = get_data_path("test_metadata.txt")
        in_table = qiime2.Artifact.load(table_f)
        in_metadata = qiime2.Metadata(
            pd.read_table(metadata_f, index_col=0))

        res = gradient_clustering(in_table, in_metadata.get_category('x'))
        res_clust = res.clustering._view(TreeNode)
        exp_str = '((o1:0.5,o2:0.5):0.5,(o3:0.5,o4:0.5):0.5);\n'
        self.assertEqual(exp_str, str(res_clust))


if __name__ == '__main__':
    unittest.main()
