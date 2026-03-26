from typing import Any, ClassVar, override

import alpha_research_framework.cross_section as xs
import alpha_research_framework.features as features
from alpha_research_framework.class_var_validator import ClassVarValidator

from .factor import Factor
from .factor_cache import FactorCache


class PrimitiveFactor(Factor, ClassVarValidator, abstract=True):
    """
    Abstract base class for factors whose values are read directly from the
    cross-section with automatic subclass validation.

    Any concrete subclass must define:
    - `FEATURE`: `type[Feature]`: feature extracted from the cross-section
    """

    FEATURE: ClassVar[type[features.Feature]]

    def __init_subclass__(cls, abstract: bool = False, **kwargs: Any) -> None:
        """
        If `abstract=False` asserts definition and type of `FEATURE` and creates
        `REQUIRED_FEATURES` as a single-element set containing it.
        """

        if not abstract:
            cls.assert_class_var_subtype("FEATURE", base_type=features.Feature)
            cls.REQUIRED_FEATURES = {cls.FEATURE}

        kwargs["abstract"] = abstract
        super().__init_subclass__(**kwargs)

    @classmethod
    @override
    def compute(cls, x: xs.CrossSection, cache: FactorCache) -> xs.Array:
        """
        Return the raw data corresponding to `cls.FEATURE` from `x`.
        
        Raises a `ValueError` if the required feature is not present in the
        cross-section.
        """

        try:
            return cls._compute(x)
        except KeyError as e:
            raise ValueError(
                f"Primitive factor {cls.__name__} could not be computed: cross-"
                f"section missing required feature {cls.FEATURE.__name__}"
            ) from e
    
    @classmethod
    def _compute(cls, x: xs.CrossSection) -> xs.Array:
        return x[cls.FEATURE]
