from typing import override

import numpy as np

import alpha_research_framework.market_data as md
from alpha_research_framework.features.feature import Feature
from alpha_research_framework.features.features import Features


class LogPrice(Feature):

    ID = "log_price"
    TAG = Feature.Tag.PREDICTOR
    DEPENDENCIES = set()

    @classmethod
    @override
    def compute(
        cls,
        market_data: md.MarketData,
        features: Features,
        out: md.Array,
    ) -> None:
        out[:] = np.log(market_data.price)
