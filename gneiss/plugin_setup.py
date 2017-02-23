import importlib
from gneiss.util import HAVE_Q2

if HAVE_Q2:
    import qiime2.plugin
    import qiime2.sdk
    from gneiss import __version__


plugin = qiime2.plugin.Plugin(
    name='gneiss',
    version=__version__,
    website='https://biocore.github.io/gneiss/',
    package='gneiss'
)

importlib.import_module('gneiss.regression')
