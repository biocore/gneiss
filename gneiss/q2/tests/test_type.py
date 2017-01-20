# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the GPLv3 License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

import unittest

from gneiss.q2._type import (Regression, Linear, LinearMixedEffects,
                             Hierarchy, Cluster)
from q2_types.tree import NewickDirectoryFormat
from gneiss.q2._format import RegressionDirectoryFormat
from qiime.plugin.testing import TestPluginBase

class TestTypes(TestPluginBase):

    package = "gneiss.q2.tests"

    def test_regression_type_registration(self):
        self.assertRegisteredSemanticType(Regression)

    def test_regression_type_registration(self):
        self.assertRegisteredSemanticType(Linear)

    def test_regression_type_registration(self):
        self.assertRegisteredSemanticType(LinearMixedEffects)

    def test_regression_linear_lme_to_regression_fmt_registration(self):
        self.assertSemanticTypeRegisteredToFormat(
            Regression[Linear | LinearMixedEffects], RegressionDirectoryFormat)

    def test_regression_type_registration(self):
        self.assertRegisteredSemanticType(Hierarchy)

    def test_regression_type_registration(self):
        self.assertRegisteredSemanticType(Cluster)

    def test_hierarchy_cluster_to_newick_dir_fmt_registration(self):
        self.assertSemanticTypeRegisteredToFormat(
            Hierarchy[Cluster], NewickDirectoryFormat)


if __name__ == '__main__':
    unittest.main()
