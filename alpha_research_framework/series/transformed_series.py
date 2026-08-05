from collections.abc import Callable
from typing import Any, ClassVar, override

import alpha_research_framework.market_data as md

from .cache import Cache
from .series import Series


class TransformedSeries(Series, abstract=True):
    """
    Abstract base class for market series derived from transformations of other
    series.

    Should not be directly subclassed - `TransformedSeries` may be created by
    application of `transform` onto an already existing `Series`.
    """

    # Parent series
    SOURCE: ClassVar[type[Series]]

    # Function to apply to parent series' output
    TRANSFORM: ClassVar[Callable[[md.Array], None]]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """
        Asserts definition and type of `SOURCE` and definition, type and number
        of arguments of `TRANSFORM`.
        """
        
        super().__init_subclass__(**kwargs)
        cls.assert_class_var_subtype("SOURCE", base_type=Series)
        
        def expected_transform(arr: md.Array) -> None:
            ...
        cls.assert_class_var_function("TRANSFORM", prototype=expected_transform)

    @classmethod
    @override
    def compute(
        cls,
        market_data: md.MarketData,
        cache: Cache,
        allocator: md.Allocator,
    ) -> md.Array:
        """
        Return an `md.Array` populated with values resulting from the
        application of `TRANSFORM` to the output of `SOURCE`.
         
        Returns the array straight from `cache` if available else computes it
        and stores it in `cache` before returning.
        """

        try:
            return cache[cls]
        except KeyError:
            out = cls._compute(market_data, cache, allocator)
            cache[cls] = out
            return out

    @classmethod
    def _compute(
        cls,
        market_data: md.MarketData,
        cache: Cache,
        allocator: md.Allocator,
    ) -> md.Array:
        """
        Allocate and return a new `md.Array` populated with values resulting
        from the application of `TRANSFORM` to the output of `SOURCE`.
        """

        out = allocator.allocate(identifier=cls.__name__)
        out[:] = cls.SOURCE(market_data, cache, allocator)
        cls.TRANSFORM(out)
        return out
