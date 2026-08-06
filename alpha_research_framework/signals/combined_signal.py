from collections.abc import Callable
from typing import Any, ClassVar, override

import alpha_research_framework.cross_section as xs

from .cache import Cache
from .signal import Signal


class CombinedSignal(Signal):
    """
    Abstract base class for cross-sectional signals derived from combinations of
    other signals.
    
    Should not be directly subclassed - `CombinedSignal`s may be created by
    application of +, -, *, / between two already existing `Signal`s.
    """

    # Parent signals
    SOURCE_LEFT: ClassVar[type[Signal]]
    SOURCE_RIGHT: ClassVar[type[Signal]]

    # Function to apply to parent signals' output
    BINARY_OP: ClassVar[Callable[[xs.Array, xs.Array], xs.Array]]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """
        Asserts definition and type of `SOURCE_LEFT` and `SOURCE_RIGHT` and
        definition, type and number of arguments of `BINARY_OP`.
        """
        
        super().__init_subclass__(**kwargs)
        cls.assert_class_var_subtype("SOURCE_LEFT", base_type=Signal)
        cls.assert_class_var_subtype("SOURCE_RIGHT", base_type=Signal)

        def expected_binary_op(left: xs.Array, right: xs.Array) -> xs.Array:
            ...
        cls.assert_class_var_function("BINARY_OP", prototype=expected_binary_op)

    @classmethod
    @override
    def compute(cls, cross_section: xs.CrossSection, cache: Cache) -> xs.Array:
        """
        Return an `xs.Array` populated with values resulting from the
        application of `BINARY_OP` to the outputs of `SOURCE_LEFT` and
        `SOURCE_RIGHT`.
         
        Returns the array straight from `cache` if available else computes it
        and stores it in `cache` before returning.
        """
        
        try:
            return cache[cls]
        except KeyError:
            result = cls._compute(cross_section, cache)
            cache[cls] = result
            return result
    
    @classmethod
    def _compute(cls, cross_section: xs.CrossSection, cache: Cache) -> xs.Array:
        """
        Return a new `xs.Array` populated with values resulting from the
        application of `BINARY_OP` to the outputs of `SOURCE_LEFT` and
        `SOURCE_RIGHT`.
        """

        return cls.BINARY_OP(
            cls.SOURCE_LEFT(cross_section, cache),
            cls.SOURCE_RIGHT(cross_section, cache),
        )
