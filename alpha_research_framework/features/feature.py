from abc import ABC, abstractmethod
from typing import ParamSpec

from alpha_research_framework.features.feature_error import FeatureError
from alpha_research_framework.features.feature_spec import FeatureSpec
from alpha_research_framework.features.feature_tag import FeatureTag
from alpha_research_framework.features.features import Features
from alpha_research_framework.market_data_view import (
    MarketArray,
    MarketDataView,
)


class Feature(ABC):
    """Abstract feature with automatic dependency checking."""

    TAG: FeatureTag

    def __init__(self) -> None:
        self._name: str
        self._dependencies: set[FeatureSpec] = set()

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        cls._wrap_init()
        cls._wrap_compute()

    @property
    def name(self) -> str:
        return self._name
    
    @property
    def dependencies(self) -> set[FeatureSpec]:
        return self._dependencies

    @abstractmethod
    def compute(
        self,
        market_data: MarketDataView,
        features: Features,
        out: MarketArray
    ) -> None:
        """
        Populate out with values calculated from raw market data and/or
        already computed features.
        """
        ...

    @classmethod
    def _wrap_init(cls) -> None:
        """
        Wrap the __init__ method with a check to ensure subclasses cannot depend
        on features with a higher TAG.
        """

        original = cls.__init__

        P = ParamSpec("P")
        def wrapped(self: Feature, *args: P.args, **kwargs: P.kwargs) -> None:
            original(self, *args, **kwargs)
            for feature in self._dependencies:
                if feature.tag > self.TAG:
                    raise FeatureError(
                        f"Feature {self._name} cannot be instantiated: it "
                        f"cannot depend on {feature.name} with higher TAG."
                    )

        cls.__init__ = wrapped

    @classmethod
    def _wrap_compute(cls) -> None:
        """Ensure all dependencies are present in the given features."""

        original = getattr(cls, "compute", None)
        if original is None or getattr(original, "__isabstractmethod__", False):
            return

        def wrapper(
            self: Feature,
            market_data: MarketDataView,
            features: Features,
            out: MarketArray
        ) -> None:
            missing = self._dependencies - features.keys()
            if missing:
                raise FeatureError(
                    f"Feature {self._name} cannot be computed: missing "
                    f"dependencies {missing}."
                )
            return original(self, market_data, features, out)

        setattr(cls, "compute", wrapper)
