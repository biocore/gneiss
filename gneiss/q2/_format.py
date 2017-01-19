# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the GPLv3 License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------
import qiime.plugin.model as model

from gneiss.regression._model import RegressionModel


class RegressionFormat(model.BinaryFileFormat):
    def sniff(self):
        try:
            res = RegressionModel.read_pickle(open(str(self)))
            return isinstance(res, RegressionModel)
        except:
            return False


RegressionDirectoryFormat = model.SingleFileDirectoryFormat(
    'RegressionDirectoryFormat', 'regression.pickle', RegressionFormat)
