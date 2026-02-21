from functools import cached_property

import numpy as np

import alpha_research_framework.market_data as md
from alpha_research_framework.features.feature import Feature
from alpha_research_framework.features.feature_spec import FeatureSpec
from alpha_research_framework.features.features import Features
from alpha_research_framework.features.log_price import LogPrice
from alpha_research_framework.window import Window


class FutureReturns(Feature):

    TAG = Feature.Tag.TARGET

    def __init__(self, horizon: Window) -> None:
        super().__init__()
        self._horizon = horizon

    @cached_property
    def name(self) -> str:
        return f"fut_ret_{self._horizon.value}d"
    
    def _init_dependencies(self) -> set[FeatureSpec]:
        self._log_price = FeatureSpec(LogPrice)
        return {self._log_price}

    def compute(
        self,
        market_data: md.MarketData,
        features: Features,
        out: md.Array
    ) -> None:
        """r_t = log(p_{t+horizon}) - log(p_t)"""

        horizon = self._horizon.value
        log_price = features[self._log_price]
        out[:-horizon] = log_price[horizon:] - log_price[:-horizon]
        out[-horizon:] = np.nan
