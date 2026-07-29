from __future__ import annotations

from abc import abstractmethod
from enum import IntEnum
from typing import Any, ClassVar

import alpha_research_framework.market_data as md
from alpha_research_framework.class_var_validator import ClassVarValidator
from alpha_research_framework.operator import Operator

from .feature_cache import FeatureCache


class Feature(Operator, ClassVarValidator):
    """
    Abstract base class for market features with automatic subclass validation.

    Should not be directly subclassed - features may be created by subclassing
    `PrimitiveFeature` or by transforming `Feature`s via application of
    subclasses of `Transform` resulting in a `DerivedFeature`.
    """

    class Tag(IntEnum):
        PREDICTOR = 0
        TARGET = 1

    TAG: ClassVar[Tag]

    def __init_subclass__(cls, abstract: bool = False, **kwargs: Any) -> None:
        """If `abstract=False` asserts definition and type of `TAG`."""
        
        super().__init_subclass__(**kwargs)
        if not abstract:
            cls.assert_class_var("TAG", type=Feature.Tag)

    @classmethod
    @abstractmethod
    def compute(
        cls,
        market_data: md.MarketData,
        cache: FeatureCache,
        out: md.Array,
    ) -> None:
        """
        Populate `out` with raw market feature value per stock per timestamp.

        Values are calculated from raw `market_data` and/or already computed
        features stored in `cache`.
        """
        ...
