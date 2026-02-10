from typing import TypeAlias

from alpha_research_framework.features.feature_spec import FeatureSpec
from alpha_research_framework.market_data_view import MarketArray

Features: TypeAlias = dict[FeatureSpec, MarketArray]
