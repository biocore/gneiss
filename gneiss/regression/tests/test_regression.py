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
import numpy as np
import pandas as pd
import pandas.util.testing as pdt
from skbio.util import get_data_path
from gneiss.regression._regression import lme_regression, ols_regression
from gneiss.regression.tests.test_ols import TestOLS
from gneiss.regression.tests.test_mixedlm import TestMixedLM
from qiime2.metadata import Metadata
import qiime2


class TestOLSPlugin(TestOLS):

    def test_ols_regression(self):
        m = Metadata(self.metadata2)
        ols_regression(self.results, self.table2, self.tree, m, 'real')

        res_coef = pd.read_csv(os.path.join(self.results, 'coefficients.csv'),
                               index_col=0)
        res_resid = pd.read_csv(os.path.join(self.results, 'residuals.csv'),
                                index_col=0)

        exp_coef = pd.DataFrame(
            {'Intercept': [1.00, 0],
             'real': [0, 1.0]},
            index=['Y1', 'Y2'])
        pdt.assert_frame_equal(res_coef, exp_coef,
                               check_exact=False,
                               check_less_precise=True)

        # Double check to make sure residuals are zero
        exp_resid = pd.DataFrame(
            [[0., 0.], [0., 0.], [0., 0.],
             [0., 0.], [0., 0.], [0., 0.],
             [0., 0.], [0., 0.], [0., 0.],
             [0., 0.], [0., 0.], [0., 0.],
             [0., 0.], [0., 0.], [0., 0.]],
            index=['s1', 's2', 's3', 's4', 's5',
                   's6', 's7', 's8', 's9', 's10',
                   's11', 's12', 's13', 's14', 's15'],
            columns=['Y1', 'Y2'])
        exp_resid = exp_resid.sort_index()
        res_resid = res_resid.sort_index()
        pdt.assert_frame_equal(exp_resid, res_resid)

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

        res_coef = pd.read_csv(os.path.join('regression_summary_dir',
                                            'coefficients.csv'),
                               index_col=0)

        self.assertAlmostEqual(res_coef.loc['y0', 'ph'],
                               0.356690, places=5)
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
