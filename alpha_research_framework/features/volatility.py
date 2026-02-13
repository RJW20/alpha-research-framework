from alpha_research_framework.features.feature import Feature
from alpha_research_framework.features.feature_spec import FeatureSpec
from alpha_research_framework.features.feature_tag import FeatureTag
from alpha_research_framework.features.features import Features
from alpha_research_framework.features.returns import Returns
from alpha_research_framework.market_data_view import (
    MarketArray,
    MarketDataView,
)
from alpha_research_framework.window import Window


class Volatility(Feature):

    NAME: str = "vol"
    TAG = FeatureTag.PREDICTOR

    def __init__(self, lookback: Window) -> None:
        super().__init__()
        self._name = f"{self.NAME}_{lookback.value}d"
        self._returns_1d = FeatureSpec(Returns, Window.DAY)
        self._dependencies = {self._returns_1d}
        self._lookback = lookback

    def compute(
        self,
        market_data: MarketDataView,
        features: Features,
        out: MarketArray
    ) -> None:
        """s_t = sqrt(sum(r_{t-lookback+1}, ..., r_{t}) / (lookback - 1))"""

        lookback = self._lookback.value
        returns_1d = features[self._returns_1d]
        self._rolling_std(returns_1d, lookback, out)
