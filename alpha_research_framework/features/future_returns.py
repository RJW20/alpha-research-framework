import numpy as np

from alpha_research_framework.features.feature import Feature
from alpha_research_framework.features.feature_spec import FeatureSpec
from alpha_research_framework.features.feature_tag import FeatureTag
from alpha_research_framework.features.features import Features
from alpha_research_framework.features.log_price import LogPrice
from alpha_research_framework.market_data_view import (
    MarketArray,
    MarketDataView,
)
from alpha_research_framework.window import Window


class FutureReturns(Feature):

    NAME: str = "fut_ret"
    TAG = FeatureTag.TARGET
    _ENTRY_LAG: int = 1

    def __init__(self, horizon: Window) -> None:
        super().__init__()
        self._name = f"{self.NAME}_{horizon.value}d"
        self._log_price_dependency = FeatureSpec(LogPrice)
        self._dependencies = {self._log_price_dependency}
        self.horizon = horizon

    def compute(
        self,
        market_data: MarketDataView,
        features: Features,
        out: MarketArray
    ) -> None:
        """r_t = log(p_{t+entry_lag+horizon} / p_{t+entry_lag})"""

        start = self._ENTRY_LAG
        end = self._ENTRY_LAG + self.horizon.value
        log_price = features[self._log_price_dependency]
        out[:-end] = log_price[end:] - log_price[start:-end + start]
        out[-end:] = np.nan
