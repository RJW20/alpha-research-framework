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
    
    _ENTRY_LAG: int = 1

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
        """r_t = log(p_{t+entry_lag+horizon} / p_{t+entry_lag})"""

        start = self._ENTRY_LAG
        end = self._ENTRY_LAG + self._horizon.value
        log_price = features[self._log_price]
        out[:-end] = log_price[end:] - log_price[start:-end + start]
        out[-end:] = np.nan
