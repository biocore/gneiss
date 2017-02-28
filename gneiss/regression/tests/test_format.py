# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the GPLv3 License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------
import shutil
import unittest
import os
from gneiss.regression._format import (RegressionFormat_g,
                                       RegressionDirectoryFormat_g)
from qiime2.plugin.testing import TestPluginBase


class TestFormats(TestPluginBase):
    package = "gneiss.regression.tests"

    def test_regression_format_validate_positive(self):
        filepath = self.get_data_path('ols.pickle')
        format = RegressionFormat_g(filepath, mode='r')

        format.validate()

    def test_regression_format_validate_negative(self):
        filepath = self.get_data_path('not-regression.pickle')
        format = RegressionFormat_g(filepath, mode='r')

        with self.assertRaisesRegex(ValueError, 'RegressionFormat_g'):
            format.validate()

    def test_regression_directory_format_validate_positive(self):
        filepath = self.get_data_path('ols.pickle')
        shutil.copy(filepath, os.path.join(self.temp_dir.name,
                                           'regression.pickle'))
        format = RegressionDirectoryFormat_g(self.temp_dir.name, mode='r')
        format.validate()


if __name__ == '__main__':
    unittest.main()
