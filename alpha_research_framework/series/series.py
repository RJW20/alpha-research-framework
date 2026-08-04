from __future__ import annotations

from abc import abstractmethod
from enum import IntEnum
from typing import Any, ClassVar

import alpha_research_framework.market_data as md
from alpha_research_framework.class_var_validator import ClassVarValidator
from alpha_research_framework.operator import Operator

from .cache import Cache


class Series(Operator, ClassVarValidator):
    """
    Abstract base class for market series with automatic subclass validation.

    Should not be directly subclassed - series may be created by subclassing
    `ObservableSeries` or by transforming already existing `Series` via
    application of `transform` resulting in a `TransformedSeries`.
    """

    class Tag(IntEnum):
        PREDICTOR = 0
        TARGET = 1

    TAG: ClassVar[Tag]

    def __init_subclass__(cls, abstract: bool = False, **kwargs: Any) -> None:
        """If `abstract=False` asserts definition and type of `TAG`."""
        
        super().__init_subclass__(**kwargs)
        if not abstract:
            cls.assert_class_var("TAG", type=Series.Tag)

    @classmethod
    @abstractmethod
    def compute(
        cls,
        market_data: md.MarketData,
        cache: Cache,
        allocator: md.Allocator,
    ) -> md.Array:
        """
        Return an `md.Array` with raw market series values per stock per
        timestamp.

        Values are calculated from raw `market_data` and/or already computed
        series stored in `cache`.
        """
        ...
