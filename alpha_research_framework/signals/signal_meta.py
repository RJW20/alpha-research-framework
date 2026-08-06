from __future__ import annotations

import operator
from collections.abc import Callable
from typing import TYPE_CHECKING

import alpha_research_framework.cross_section as xs
from alpha_research_framework.operator import OperatorMeta

if TYPE_CHECKING:
    from .combined_signal import CombinedSignal
    from .negated_signal import NegatedSignal
    from .signal import Signal


class SignalMeta(OperatorMeta):
    """
    Metaclass for cross-sectional signals.
    
    Provides utility for combining or negating signals.
    """

    def __add__(cls, other: type[Signal]) -> type[CombinedSignal]:              # noqa: N805
        return cls._combine(
            other,
            operator.add,
            f"({cls.__name__}Plus{other.__name__})",
        )
    
    def __sub__(cls, other: type[Signal]) -> type[CombinedSignal]:              # noqa: N805
        return cls._combine(
            other,
            operator.sub,
            f"({cls.__name__}Minus{other.__name__})",
        )
    
    def __mul__(cls, other: type[Signal]) -> type[CombinedSignal]:              # noqa: N805
        return cls._combine(
            other,
            operator.mul,
            f"({cls.__name__}Times{other.__name__})",
        )
    
    def __truediv__(cls, other: type[Signal]) -> type[CombinedSignal]:          # noqa: N805
        return cls._combine(
            other,
            operator.truediv,
            f"({cls.__name__}Divide{other.__name__})",
        )
    
    def __neg__(cls) -> type[NegatedSignal]:                                    # noqa: N805
        return cls._negate(f"(Minus{cls.__name__})")
    
    def _combine(
        cls,                                                                    # noqa: N805
        other: type[Signal],
        op: Callable[[xs.Array, xs.Array], xs.Array],
        name: str,
    ) -> type[CombinedSignal]:
        """
        Return a dynamically generated concrete subclass of `CombinedSignal`.
        
        The subclass' `SOURCE_LEFT` is `cls` and `SOURCE_RIGHT` is `other`.
        The subclass' `BINARY_OP` is `op`.
        """

        from .combined_signal import CombinedSignal
        from .signal import Signal

        if not issubclass(other, Signal):                                       # type: ignore
            raise TypeError(
                f"Unable to build new signal: {other.__name__} must be a "
                "subclass of Signal"
            )

        return type(
            name,
            (CombinedSignal,),
            {
                "SOURCE_LEFT": cls,
                "SOURCE_RIGHT": other,
                "BINARY_OP": op,
            }
        )
    
    def _negate(cls, name: str) -> type[NegatedSignal]:                         # noqa: N805
        """
        Return a dynamically generated concrete subclass of `NegatedSignal`.
        
        The subclass' `SOURCE` is `cls`'.
        """

        from .negated_signal import NegatedSignal

        return type(
            name,
            (NegatedSignal,),
            {
                "SOURCE": cls,
            },
        )
