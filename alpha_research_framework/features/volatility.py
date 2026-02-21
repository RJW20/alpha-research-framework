from functools import cached_property

import alpha_research_framework.market_data as md
from alpha_research_framework.features.feature import Feature
from alpha_research_framework.features.feature_spec import FeatureSpec
from alpha_research_framework.features.features import Features
from alpha_research_framework.features.returns import Returns
from alpha_research_framework.window import Window


class Volatility(Feature):

    TAG = Feature.Tag.PREDICTOR

    def __init__(self, lookback: Window) -> None:
        super().__init__()
        self._lookback = lookback

    @cached_property
    def name(self) -> str:
        return f"vol_{self._lookback.value}d"
    
    def _init_dependencies(self) -> set[FeatureSpec]:
        self._ret_1d = FeatureSpec(Returns, Window.DAY)
        return {self._ret_1d}

    def compute(
        self,
        market_data: md.MarketData,
        features: Features,
        out: md.Array
    ) -> None:
        """s_t = sqrt(sum(r_{t-lookback+1}, ..., r_{t}) / (lookback - 1))"""

        lookback = self._lookback.value
        ret_1d = features[self._ret_1d]
        self._rolling_std(ret_1d, lookback, out)
