from typing import Any, ClassVar, override

import alpha_research_framework.cross_section as xs
import alpha_research_framework.series as series
from alpha_research_framework.class_var_validator import ClassVarValidator

from .cache import Cache
from .signal import Signal


class SeriesSignal(Signal, ClassVarValidator):
    """
    Abstract base class for cross-sectional signals whose values are read
    directly from the cross-section with automatic subclass validation.

    Any concrete subclass must define:
    - `SERIES`: `type[series.Series]`: series extracted from the cross-section
    """

    SERIES: ClassVar[type[series.Series]]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Asserts definition and type of `SERIES`."""

        super().__init_subclass__(**kwargs)
        cls.assert_class_var_subtype("SERIES", base_type=series.Series)

    @classmethod
    @override
    def compute(cls, cross_section: xs.CrossSection, cache: Cache) -> xs.Array:
        """
        Return a reference to the `xs.Array` pertaining to `SERIES` straight
        from `cross_section`.
        
        Raises a `ValueError` if the required series is not present in the
        cross-section.
        """

        try:
            return cls._compute(cross_section)
        except KeyError as e:
            raise ValueError(
                f"Cross-sectional series {cls.__name__} could not be computed: "
                f"cross-section missing required series {cls.SERIES.__name__}"
            ) from e
    
    @classmethod
    def _compute(cls, cross_section: xs.CrossSection) -> xs.Array:
        return cross_section[cls.SERIES]
