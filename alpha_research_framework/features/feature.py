from abc import ABC, abstractmethod
from typing import Callable

from alpha_research_framework.features.feature_spec import FeatureSpec
from alpha_research_framework.features.feature_tag import FeatureTag
from alpha_research_framework.features.features import Features
from alpha_research_framework.market_array import MarketArray
from alpha_research_framework.universe import MarketData


class Feature(ABC):
    """Abstract feature with automatic dependency checking in compute."""

    TAG: FeatureTag

    def __init__(self) -> None:
        self.name: str
        self.dependencies: set[FeatureSpec] = set()

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        original = getattr(cls, "compute", None)
        if original is None or getattr(original, "__isabstractmethod__", False):
            return

        def wrapper(
            self: Feature,
            market_data: MarketData,
            features: Features,
            out: MarketArray
        ) -> Callable[[Feature, MarketData, Features, MarketData], None]:
            missing = self.dependencies - features.keys()
            if missing:
                raise ValueError(
                    f"Feature {self.name} cannot be computed: missing "
                    f"dependencies {missing}"
                )
            return original(self, market_data, features, out)

        setattr(cls, "compute", wrapper)

    @abstractmethod
    def compute(
        self,
        market_data: MarketData,
        features: Features,
        out: MarketArray
    ) -> None:
        """
        Populate out with values calculated from raw market data and/or
        already computed features.
        """
        ...
