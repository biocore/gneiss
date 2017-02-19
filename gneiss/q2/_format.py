# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the GPLv3 License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------
import qiime.plugin.model as model

from gneiss.regression._model import RegressionModel
from gneiss.plugin_setup import plugin


class RegressionFormat_g(model.BinaryFileFormat):
    def sniff(self):
        try:
            res = RegressionModel.read_pickle(open(str(self)))
            return isinstance(res, RegressionModel)
        except:
            return False


RegressionDirectoryFormat_g = model.SingleFileDirectoryFormat(
    'RegressionDirectoryFormat_g', 'regression.pickle', RegressionFormat_g)

plugin.register_formats(
    RegressionDirectoryFormat_g
)
