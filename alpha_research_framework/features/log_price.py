from functools import cached_property

import numpy as np

import alpha_research_framework.market_data as md
from alpha_research_framework.features.feature import Feature
from alpha_research_framework.features.feature_spec import FeatureSpec
from alpha_research_framework.features.features import Features


class LogPrice(Feature):

    TAG = Feature.Tag.PREDICTOR

    @cached_property
    def name(self) -> str:
        return "log_price"
    
    def _init_dependencies(self) -> set[FeatureSpec]:
        return set()

    def compute(
        self,
        market_data: md.MarketData,
        features: Features,
        out: md.Array
    ) -> None:
        out[:] = np.log(market_data.price)
