# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------
import os
import shutil
import unittest

import pandas as pd
from skbio.util import get_data_path
from gneiss.regression.tests.test_ols import TestOLS
from gneiss.regression.tests.test_mixedlm import TestMixedLM
import qiime2


class TestOLSPlugin(TestOLS):

    def test_ols_artifact(self):
        from qiime2.plugins.gneiss.visualizers import ols_regression

        table_f = get_data_path("ols_balances.qza")
        tree_f = get_data_path("ols_tree.qza")
        metadata_f = get_data_path("test_ols_metadata.txt")

        in_table = qiime2.Artifact.load(table_f)
        in_tree = qiime2.Artifact.load(tree_f)
        in_metadata = qiime2.Metadata(
            pd.read_table(metadata_f, index_col=0))

        viz = ols_regression(in_table, in_tree, in_metadata, 'ph')
        viz.visualization.export_data('regression_summary_dir')

        # check coefficient
        res_coef = pd.read_csv(os.path.join('regression_summary_dir',
                                            'coefficients.csv'),
                               index_col=0)
        self.assertAlmostEqual(res_coef.loc['y0', 'ph'],
                               0.356690, places=5)
        # check pvalue
        res_pvalue = pd.read_csv(os.path.join('regression_summary_dir',
                                              'pvalues.csv'),
                                 index_col=0)
        self.assertAlmostEqual(res_pvalue.loc['y0', 'ph'],
                               1.59867977447e-06, places=5)

        # check balance
        res_balance = pd.read_csv(os.path.join('regression_summary_dir',
                                               'balances.csv'),
                                  index_col=0)
        self.assertAlmostEqual(res_balance.loc['y0'][0],
                               -0.756213598577, places=5)

        # check residual
        res_resid = pd.read_csv(os.path.join('regression_summary_dir',
                                             'residuals.csv'),
                                index_col=0)
        self.assertAlmostEqual(res_resid.loc['y0'][0], -0.164646694173,
                               places=5)

        # check predicted
        res_pred = pd.read_csv(os.path.join('regression_summary_dir',
                                            'predicted.csv'),
                               index_col=0)
        self.assertAlmostEqual(res_pred.loc['y0'][0],
                               -0.591566904404, places=5)

        shutil.rmtree('regression_summary_dir')


class TestMixedLMPlugin(TestMixedLM):

    def test_lme_artifact(self):
        from qiime2.plugins.gneiss.visualizers import lme_regression

        table_f = get_data_path("lme_balances.qza")
        tree_f = get_data_path("lme_tree.qza")
        metadata_f = get_data_path("test_lme_metadata.txt")

        in_table = qiime2.Artifact.load(table_f)
        in_tree = qiime2.Artifact.load(tree_f)
        in_metadata = qiime2.Metadata(
            pd.read_table(metadata_f, index_col=0))

        viz = lme_regression(in_table, in_tree, in_metadata,
                             'ph', 'host_subject_id')
        viz.visualization.export_data('regression_summary_dir')

        res_coef = pd.read_csv(os.path.join('regression_summary_dir',
                                            'coefficients.csv'),
                               index_col=0)

        self.assertAlmostEqual(res_coef.loc['y0', 'groups RE'],
                               1.105630e+00, places=5)


if __name__ == '__main__':
    unittest.main()
