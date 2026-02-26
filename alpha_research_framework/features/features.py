from typing import TYPE_CHECKING, TypeAlias

import alpha_research_framework.market_data as md

if TYPE_CHECKING:
    from alpha_research_framework.features.feature import Feature

Features: TypeAlias = dict[type["Feature"], md.Array]
