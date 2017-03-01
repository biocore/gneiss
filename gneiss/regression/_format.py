# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the GPLv3 License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------
import qiime2.plugin.model as model
from gneiss.plugin_setup import plugin
from gneiss.regression._ols import OLSModel
from gneiss.regression._mixedlm import LMEModel


class LinearRegressionFormat_g(model.BinaryFileFormat):
    def sniff(self):
        try:
            # just check if the file is pickleable.
            OLSModel.read_pickle(str(self))
            return True
        except:
            return False


LinearRegressionDirectoryFormat_g = model.SingleFileDirectoryFormat(
    'LinearRegressionDirectoryFormat_g', 'ols_regression.pickle',
    LinearRegressionFormat_g)


plugin.register_formats(
    LinearRegressionFormat_g,
    LinearRegressionDirectoryFormat_g
)


class LinearMixedEffectsFormat_g(model.BinaryFileFormat):
    def sniff(self):
        try:
            # just check if the file is pickleable.
            LMEModel.read_pickle(str(self))
            return True
        except:
            return None


LinearMixedEffectsDirectoryFormat_g = model.SingleFileDirectoryFormat(
    'LinearMixedEffectsDirectoryFormat_g', 'lme_regression.pickle',
    LinearMixedEffectsFormat_g)


plugin.register_formats(
    LinearMixedEffectsFormat_g,
    LinearMixedEffectsDirectoryFormat_g
)
