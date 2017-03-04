# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------
from ._ols import OLSModel
from ._mixedlm import LMEModel
from ._format import (LinearRegressionFormat_g,
                      LinearMixedEffectsFormat_g)
from gneiss.plugin_setup import plugin


# linear regression
@plugin.register_transformer
def _1(data: OLSModel) -> LinearRegressionFormat_g:
    ff = LinearRegressionFormat_g()
    with ff.open() as fh:
        data.write_pickle(fh)
    return ff


@plugin.register_transformer
def _2(ff: LinearRegressionFormat_g) -> OLSModel:
    with ff.open() as fh:
        return OLSModel.read_pickle(fh)


# linear mixed effects
@plugin.register_transformer
def _3(data: LMEModel) -> LinearMixedEffectsFormat_g:
    ff = LinearMixedEffectsFormat_g()
    with ff.open() as fh:
        data.write_pickle(fh)
    return ff


@plugin.register_transformer
def _4(ff: LinearMixedEffectsFormat_g) -> LMEModel:
    with ff.open() as fh:
        return LMEModel.read_pickle(fh)
