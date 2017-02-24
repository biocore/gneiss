from ._ols import OLSModel
from ._mixedlm import LMEModel
from ._format import RegressionFormat_g
from gneiss.plugin_setup import plugin


@plugin.register_transformer
def _1(data: OLSModel) -> RegressionFormat_g:
    ff = RegressionFormat_g()
    with ff.open() as fh:
        data.write_pickle(fh)
    return ff


@plugin.register_transformer
def _2(ff: RegressionFormat_g) -> OLSModel:
    with ff.open() as fh:
        return fh.read_pickle(fh)


@plugin.register_transformer
def _3(data: LMEModel) -> RegressionFormat_g:
    ff = RegressionFormat_g()
    with ff.open() as fh:
        data.write_pickle(fh)
    return ff


@plugin.register_transformer
def _4(ff: RegressionFormat_g) -> LMEModel:
    with ff.open() as fh:
        return fh.read_pickle(fh)
