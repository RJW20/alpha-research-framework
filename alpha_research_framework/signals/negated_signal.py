from typing import Any, ClassVar, override

import alpha_research_framework.cross_section as xs

from .cache import Cache
from .signal import Signal


class NegatedSignal(Signal):
    """
    Abstract base class for cross-sectional signals derived from negating other
    signals.
    
    Should not be directly subclassed - `CombinedSignal`s may be created by
    application of +, -, *, / between two already existing `Signal`s.
    """

    # Parent signal
    SOURCE: ClassVar[type[Signal]]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Asserts definition and type of `SOURCE`."""
        
        super().__init_subclass__(**kwargs)
        cls.assert_class_var_subtype("SOURCE", base_type=Signal)

    @classmethod
    @override
    def compute(cls, cross_section: xs.CrossSection, cache: Cache) -> xs.Array:
        """
        Return an `xs.Array` populated with values resulting from the
        negation of the output of `SOURCE`,
         
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
        negation of the output of `SOURCE`.
        """

        return - cls.SOURCE(cross_section, cache)
