from abc import abstractmethod
from typing import Any, ClassVar, override

import alpha_research_framework.market_data as md

from .feature import Feature
from .feature_cache import FeatureCache


class DerivedFeature(Feature, abstract=True):
    """
    Abstract base class for features derived from transformations of other
    features.

    Should not be directly subclassed - derived features may be created by
    transforming `Feature`s via application of subclasses of `Transform`.
    """

    # Feature transformation has been applied to
    SOURCE: ClassVar[type[Feature]]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Asserts definition and type of `SOURCE`."""
        
        super().__init_subclass__(**kwargs)
        cls.assert_class_var_subtype("SOURCE", base_type=Feature)

    @classmethod
    @override
    def compute(
        cls,
        market_data: md.MarketData,
        cache: FeatureCache,
        out: md.Array,
    ) -> None:
        """
        Fill `out` with values pertaining to a transform of `SOURCE` straight
        from `cache` if available else computed.
        """

        try:
            out[:] = cache[cls]
        except KeyError:
            cls._compute(market_data, cache, out)

    @classmethod
    @abstractmethod
    def _compute(
        cls,
        market_data: md.MarketData,
        cache: FeatureCache,
        out: md.Array,
    ) -> None:
        """
        Fill `out` with a transform of the values pertaining to `SOURCE`.

        Configured in `TransformMeta.__call__`.
        """
        ...
