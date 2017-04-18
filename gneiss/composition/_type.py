# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------
from qiime2.plugin import SemanticType
from q2_types.feature_table import (
    FeatureTable, BIOMV100DirFmt, BIOMV210DirFmt)
from gneiss.plugin_setup import plugin

Balance = SemanticType('Balance', variant_of=FeatureTable.field['content'])

plugin.register_semantic_types(Balance)

plugin.register_semantic_type_to_format(
    FeatureTable[Balance],
    artifact_format=BIOMV210DirFmt
)

plugin.register_semantic_type_to_format(
    FeatureTable[Balance],
    artifact_format=BIOMV100DirFmt
)
