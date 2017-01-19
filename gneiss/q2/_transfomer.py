# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the GPLv3 License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------
from ..plugin_setup import plugin
from . import RegressionFormat
from gneiss.regression import RegressionModel


@plugin.register_transformer
def _1(data: RegressionModel) -> RegressionFormat:
    ff = NewickFormat()
    with ff.open() as fh:
        data.write_pickle(fh)
    return ff


@plugin.register_transformer
def _2(data: RegressionModel) -> RegressionFormat:
    with ff.open() as fh:
        data.read_pickle(fh)
