# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the GPLv3 License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------
from qiime.plugin import SemanticType
from q2_types.feature_table import (
    FeatureTable, Frequency, BIOMV100DirFmt, BIOMV210DirFmt)


# Regression types
Regression = SemanticType('Regression', field_names=['type'])

Linear = SemanticType('Linear',
                      variant_of=Regression.field['type'])
LinearMixedEffects = SemanticType('LinearMixedEffects',
                                  variant_of=Regression.field['type'])

# Tree types
Hierarchy = SemanticType('Hierarchy', field_names=['type'])
Cluster = SemanticType('Cluster', variant_of=Hierarchy.field['type'])
