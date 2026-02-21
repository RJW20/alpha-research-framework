from functools import cached_property

import numpy as np

import alpha_research_framework.market_data as md
from alpha_research_framework.features.feature import Feature
from alpha_research_framework.features.feature_spec import FeatureSpec
from alpha_research_framework.features.features import Features
from alpha_research_framework.features.log_price import LogPrice
from alpha_research_framework.window import Window


class Returns(Feature):

    TAG = Feature.Tag.PREDICTOR

    def __init__(self, lookback: Window) -> None:
        super().__init__()
        self._lookback = lookback

    @cached_property
    def name(self) -> str:
        return f"ret_{self._lookback.value}d"
    
    def _init_dependencies(self) -> set[FeatureSpec]:
        self._log_price = FeatureSpec(LogPrice)
        return {self._log_price}

    def compute(
        self,
        market_data: md.MarketData,
        features: Features,
        out: md.Array
    ) -> None:
        """r_t = log(p_t) - log(p_{t-lookback})"""

        lookback = self._lookback.value
        log_price = features[self._log_price]
        out[:lookback] = np.nan
        out[lookback:] = log_price[lookback:] - log_price[:-lookback]
