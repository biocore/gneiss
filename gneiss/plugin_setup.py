import importlib

import qiime2.plugin
import qiime2.sdk
from gneiss import __version__

_citation = ('Morton JT, Sanders J, Quinn RA, McDonald D, Gonzalez A, '
             'Vázquez-Baeza Y Navas-Molina JA, Song SJ, Metcalf JL, '
             'Hyde ER, Lladser M, Dorrestein PC Knight R. 2017. '
             'Balance trees reveal microbial niche differentiation '
             'mSystems 2:e00162-16. '
             'https://doi.org/10.1128/mSystems.00162-16.')

plugin = qiime2.plugin.Plugin(
    name='gneiss',
    version=__version__,
    website='https://biocore.github.io/gneiss/',
    citation_text=_citation,
    package='gneiss')

importlib.import_module('gneiss.regression')
importlib.import_module('gneiss.plot')
