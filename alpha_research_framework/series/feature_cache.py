from typing import TYPE_CHECKING, TypeAlias

import alpha_research_framework.market_data as md

if TYPE_CHECKING:
    from .feature import Feature

FeatureCache: TypeAlias = dict[type["Feature"], md.Array]
