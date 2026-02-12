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


class Returns(Feature):

    NAME: str = "ret"
    TAG = FeatureTag.PREDICTOR

    def __init__(self, lookback: Window) -> None:
        super().__init__()
        self._name = f"{self.NAME}_{lookback.value}d"
        self._log_price_dependency = FeatureSpec(LogPrice)
        self._dependencies = {self._log_price_dependency}
        self._lookback = lookback

    def compute(
        self,
        market_data: MarketDataView,
        features: Features,
        out: MarketArray
    ) -> None:
        """r_t = log(p_t) - log(p_{t-lookback})"""

        lookback = self._lookback.value
        log_price = features[self._log_price_dependency]
        out[:lookback] = np.nan
        out[lookback:] = log_price[lookback:] - log_price[:-lookback]
