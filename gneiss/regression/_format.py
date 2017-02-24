# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the GPLv3 License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------
import qiime2.plugin.model as model
from gneiss.plugin_setup import plugin
import pickle


class RegressionFormat_g(model.BinaryFileFormat):
    def sniff(self):
        return True
        try:
            # just check if the file is pickleable.
            pickle.load(open(str(self)))
            return True
        except:
            return False

RegressionDirectoryFormat_g = model.SingleFileDirectoryFormat(
    'RegressionDirectoryFormat_g', 'regression.pickle', RegressionFormat_g)


plugin.register_formats(
    RegressionFormat_g,
    RegressionDirectoryFormat_g
)
