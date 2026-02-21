from typing import TypeAlias

import alpha_research_framework.market_data as md
from alpha_research_framework.features.feature_spec import FeatureSpec

Features: TypeAlias = dict[FeatureSpec, md.Array]
