import numpy as np

from alpha_research_framework.features.feature import Feature
from alpha_research_framework.features.feature_tag import FeatureTag
from alpha_research_framework.features.features import Features
from alpha_research_framework.market_data_view import (
    MarketArray,
    MarketDataView,
)


class LogPrice(Feature):

    NAME: str = "log_price"
    TAG = FeatureTag.PREDICTOR

    def __init__(self) -> None:
        super().__init__()
        self.name = f"{self.NAME}"

    def compute(
        self,
        market_data: MarketDataView,
        features: Features,
        out: MarketArray
    ) -> None:
        out[:] = np.log(market_data.price)
