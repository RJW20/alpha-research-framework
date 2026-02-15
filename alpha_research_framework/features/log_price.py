from functools import cached_property

import numpy as np

from alpha_research_framework.features.feature import Feature
from alpha_research_framework.features.feature_spec import FeatureSpec
from alpha_research_framework.features.features import Features
from alpha_research_framework.market_data_view import (
    MarketArray,
    MarketDataView,
)


class LogPrice(Feature):

    TAG = Feature.Tag.PREDICTOR

    @cached_property
    def name(self) -> str:
        return "log_price"
    
    def _init_dependencies(self) -> set[FeatureSpec]:
        return set()

    def compute(
        self,
        market_data: MarketDataView,
        features: Features,
        out: MarketArray
    ) -> None:
        out[:] = np.log(market_data.price)
