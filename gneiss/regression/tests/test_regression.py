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
import pandas.util.testing as pdt
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
        os.mkdir('regression_summary_dir')
        viz.visualization.export_data('regression_summary_dir')

        # check coefficient
        res_coef = pd.read_csv(os.path.join('regression_summary_dir',
                                            'coefficients.csv'),
                               index_col=0)
        exp_coef = pd.read_csv(get_data_path('coefficients.csv'), index_col=0)
        pdt.assert_frame_equal(res_coef.sort_index(), exp_coef.sort_index())

        # check pvalue
        res_pvalue = pd.read_csv(os.path.join('regression_summary_dir',
                                              'pvalues.csv'),
                                 index_col=0)
        res_pvalue = res_pvalue.reindex_axis(
            sorted(res_pvalue.columns), axis=1)
        exp_pvalue = pd.read_csv(get_data_path('pvalues.csv'), index_col=0)
        exp_pvalue = exp_pvalue.reindex_axis(
            sorted(exp_pvalue.columns), axis=1)

        pdt.assert_frame_equal(res_pvalue.sort_index(),
                               exp_pvalue.sort_index())

        # check residual
        res_resid = pd.read_csv(os.path.join('regression_summary_dir',
                                             'residuals.csv'),
                                index_col=0)
        res_resid = res_resid.reindex_axis(sorted(res_resid.columns), axis=1)
        exp_resid = pd.read_csv(get_data_path('residuals.csv'), index_col=0)
        exp_resid = exp_resid.reindex_axis(sorted(exp_resid.columns), axis=1)
        pdt.assert_frame_equal(res_resid.sort_index(),
                               exp_resid.sort_index())
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
        os.mkdir('regression_summary_dir')
        viz.visualization.export_data('regression_summary_dir')

        res_coef = pd.read_csv(os.path.join('regression_summary_dir',
                                            'coefficients.csv'),
                               index_col=0)

        self.assertAlmostEqual(res_coef.loc['y0', 'groups RE'],
                               1.105630e+00, places=5)
        shutil.rmtree('regression_summary_dir')


if __name__ == '__main__':
    unittest.main()
