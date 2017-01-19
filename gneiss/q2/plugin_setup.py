import qiime.plugin

import q2_gneiss
import q2_composition
# These imports are only included to support the example methods and
# visualizers. Remove these imports when you are ready to develop your plugin.

from q2_types import Phylogeny, Rooted, FeatureTable

from qiime.plugin import Plugin, Metadata, Str
from q2_gneiss._balance import ols, mixedlm
from q2_gneiss._plot import pvalplot
from q2_composition.plugin_setup import Composition
from gneiss.q2._type import BalanceModel, Linear, LinearMixedEffects
from gneiss.q2._format import RegressionDirectoryFormat, RegressionFormat


plugin = qiime.plugin.Plugin(
    name='gneiss',
    version=q2_gneiss.__version__,
    website='https://biocore.github.io/gneiss/',
    package='gneiss',
    # Information on how to obtain user support should be provided as a free
    # text string via user_support_text. If None is provided, users will
    # be referred to the plugin's website for support.
    user_support_text=None,
    # Information on how the plugin should be cited should be provided as a
    # free text string via citation_text. If None is provided, users
    # will be told to use the plugin's website as a citation.
    citation_text=None
)

# Balance Types
plugin.register_semantic_types(Balance, Regression, Linear, LinearMixedEffects)

plugin.register_formats(RegressionDirectoryFormat)
plugin.register_semantic_type_to_format(
    Regression[Linear | LinearMixedEffects],
    artifact_format=RegressionDirectoryFormat
)

plugin.methods.register_function(
    function=q2_gneiss.ols,
    inputs={'table': FeatureTable[Composition],
            'tree': Phylogeny[Rooted]},
    parameters={'metadata': Metadata},
    outputs=[('linear_model', Regression[Linear])],
    name='Linear Regression',
    description="Perform linear regression on balances."
)

plugin.methods.register_function(
    function=q2_gneiss.mixedlm,
    inputs={'table': FeatureTable[Composition], 'tree': Phylogeny[Rooted]},
    parameters={'metadata': Metadata},
    outputs=[('linear_mixed_effects_model', BalanceModel[LinearMixedEffects])],
    name='Linear mixed effects models',
    description="Perform linear mixed effects model on balances."
)

plugin.visualizers.register_function(
    function=pvalplot,
    inputs={'results': BalanceModel[Linear | LinearMixedEffects]},
    parameters={'category': Str},
    name='P-value tree plot.',
    description='Visualize Regression Coefficient Pvalues on Tree.'
)
