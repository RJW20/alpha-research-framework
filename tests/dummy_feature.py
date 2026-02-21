from functools import cached_property

import alpha_research_framework.market_data as md
from alpha_research_framework.features import Feature, Features, FeatureSpec


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
        market_data: md.MarketData,
        features: Features,
        out: md.Array
    ) -> None:
        pass
