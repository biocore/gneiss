from qiime.plugin import SemanticType
from q2_types.feature_table import (
    FeatureTable, Frequency, BIOMV100DirFmt, BIOMV210DirFmt)


Balance = SemanticType('Balance', variant_of=FeatureTable.field['content'])

Regression = SemanticType('Regression', field_names=['type'])

Linear = SemanticType('Linear',
                      variant_of=BalanceModel.field['type'])
LinearMixedEffects = SemanticType('LinearMixedEffects',
                                  variant_of=BalanceModel.field['type'])
