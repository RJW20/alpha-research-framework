from collections.abc import Callable
from typing import Any, ClassVar, override

import alpha_research_framework.market_data as md

from .cache import Cache
from .series import Series


class CombinedSeries(Series, abstract=True):
    """
    Abstract base class for market series derived from combinations of other
    series.

    Should not be directly subclassed - `CombinedSeries` may be created by
    application of +, -, *, / between two already existing `Series`.
    """

    # Parent series
    SOURCE_LEFT: ClassVar[type[Series]]
    SOURCE_RIGHT: ClassVar[type[Series]]

    # Function to apply to parent series' output
    BINARY_OP: ClassVar[Callable[[md.Array, md.Array], md.Array]]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """
        Asserts definition and type of `SOURCE_LEFT` and `SOURCE_RIGHT` and
        definition, type and number of arguments of `BINARY_OP`.
        """
        
        super().__init_subclass__(**kwargs)
        cls.assert_class_var_subtype("SOURCE_LEFT", base_type=Series)
        cls.assert_class_var_subtype("SOURCE_RIGHT", base_type=Series)

        def expected_binary_op(left: md.Array, right: md.Array) -> md.Array:
            ...
        cls.assert_class_var_function("BINARY_OP", prototype=expected_binary_op)

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
        application of `BINARY_OP` to the outputs of `SOURCE_LEFT` and
        `SOURCE_RIGHT`.
         
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
        from the application of `BINARY_OP` to the outputs of `SOURCE_LEFT` and
        `SOURCE_RIGHT`.
        """

        out = allocator.allocate(identifier=cls.__name__)
        out[:] = cls.BINARY_OP(
            cls.SOURCE_LEFT(market_data, cache, allocator),
            cls.SOURCE_RIGHT(market_data, cache, allocator),
        )
        return out
