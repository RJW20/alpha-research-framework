from functools import cached_property

from alpha_research_framework.features import Feature, Features, FeatureSpec
from alpha_research_framework.market_data_view import (
    MarketArray,
    MarketDataView,
)


class DummyFeature(Feature):
    """Dummy feature that enables class-level dependencies for easy testing."""

    __dependencies__: set[FeatureSpec] = set()

    TAG = Feature.Tag.PREDICTOR

    @cached_property
    def name(self) -> str:
        return self.__class__.__name__
    
    def _init_dependencies(self) -> set[FeatureSpec]:
        return self.__dependencies__
    
    def compute(
        self,
        market_data: MarketDataView,
        features: Features,
        out: MarketArray
    ) -> None:
        pass
