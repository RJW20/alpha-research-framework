from __future__ import annotations

import operator
from collections.abc import Callable
from typing import TYPE_CHECKING

import alpha_research_framework.market_data as md
from alpha_research_framework.operator import OperatorMeta

if TYPE_CHECKING:
    from .combined_series import CombinedSeries
    from .series import Series


class SeriesMeta(OperatorMeta):
    """
    Metaclass for market series.
    
    Provides utility for combining series.
    """

    def __add__(cls, other: type[Series]) -> type[CombinedSeries]:              # noqa: N805
        return cls._combine(
            other,
            operator.add,
            f"({cls.__name__}_plus_{other.__name__})",
        )
    
    def __sub__(cls, other: type[Series]) -> type[CombinedSeries]:              # noqa: N805
        return cls._combine(
            other,
            operator.sub,
            f"({cls.__name__}_minus_{other.__name__})",
        )
    
    def __mul__(cls, other: type[Series]) -> type[CombinedSeries]:              # noqa: N805
        return cls._combine(
            other,
            operator.mul,
            f"({cls.__name__}_times_{other.__name__})",
        )
    
    def __truediv__(cls, other: type[Series]) -> type[CombinedSeries]:          # noqa: N805
        return cls._combine(
            other,
            operator.truediv,
            f"({cls.__name__}_divide_{other.__name__})",
        )
    
    def _combine(
        cls,                                                                    # noqa: N805
        other: type[Series],
        op: Callable[[md.Array, md.Array], md.Array],
        name: str,
    ) -> type[CombinedSeries]:
        """
        Return a dynamically generated concrete subclass of `CombinedSeries`.
        
        The subclass' `TAG` is the logical maximum of the input `Series`'
        `TAG`s.
        The subclass' `SOURCE_LEFT` is `cls` and `SOURCE_RIGHT` is `other`.
        The subclass' `BINARY_OP` is `op`.
        """

        from .combined_series import CombinedSeries
        from .series import Series

        if not issubclass(other, Series):                                       # type: ignore
            raise TypeError(
                f"Unable to build new series: {other.__name__} must be a "
                "subclass of Series"
            )

        return type(
            name,
            (CombinedSeries,),
            {
                "TAG": Series.Tag(max(cls.TAG, other.TAG)),                     # type: ignore
                "SOURCE_LEFT": cls,
                "SOURCE_RIGHT": other,
                "BINARY_OP": op,
            }
        )
