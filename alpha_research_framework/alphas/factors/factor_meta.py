from __future__ import annotations

import operator
from typing import TYPE_CHECKING, Callable

import alpha_research_framework.cross_section as xs
from alpha_research_framework.operator import OperatorMeta

from .factor_cache import FactorCache

if TYPE_CHECKING:
    from .derived_factor import DerivedFactor
    from .factor import Factor


class FactorMeta(OperatorMeta):
    """
    Metaclass for cross-sectional factors.
    
    Provides utility for combining factors.
    """

    def __add__(cls, other: type[Factor]) -> type[DerivedFactor]:               # noqa: N805
        return cls._combine(
            other,
            operator.add,
            f"({cls.__name__}Plus{other.__name__})",
        )
    
    def __sub__(cls, other: type[Factor]) -> type[DerivedFactor]:               # noqa: N805
        return cls._combine(
            other,
            operator.sub,
            f"({cls.__name__}Minus{other.__name__})",
        )
    
    def __mul__(cls, other: type[Factor]) -> type[DerivedFactor]:               # noqa: N805
        return cls._combine(
            other,
            operator.mul,
            f"({cls.__name__}Times{other.__name__})",
        )
    
    def __truediv__(cls, other: type[Factor]) -> type[DerivedFactor]:           # noqa: N805
        return cls._combine(
            other,
            operator.truediv,
            f"({cls.__name__}Divide{other.__name__})",
        )
    
    def __neg__(cls) -> type[DerivedFactor]:                                    # noqa: N805
        return cls._negate(f"(Minus{cls.__name__})")
    
    def _combine(
        cls,                                                                    # noqa: N805
        other: type[Factor],
        fn: Callable[[xs.Array, xs.Array], xs.Array],
        name: str,
    ) -> type[DerivedFactor]:
        """
        Return a dynamicly generated concrete subclass of `DerivedFactor`.
        
        The subclass' `REQUIRED_FEATURES` will be the union of `cls`' and
        `other`'s `REQUIRED_FEATURES`.
        The subclass' `compute` will be the application of `fn` to the results
        of `cls.compute` and `other.compute`.
        """

        from .derived_factor import DerivedFactor
        from .factor import Factor

        if not issubclass(other, Factor):                                       # type: ignore
            raise TypeError(
                f"Unable to build new Factor: {other.__name__} must be a "
                "subclass of Factor"
            )
        
        required_features = cls.REQUIRED_FEATURES | other.REQUIRED_FEATURES     # type: ignore[attr-defined]
        
        def _compute(
            cls_: type[DerivedFactor],
            x: xs.CrossSection,
            cache: FactorCache,
        ) -> xs.Array:
            return fn(cls.compute(x, cache), other.compute(x, cache))           # type: ignore[attr-defined]

        return type(
            name,
            (DerivedFactor,),
            {
                "REQUIRED_FEATURES": required_features,
                "_compute": classmethod(_compute),
            },
        )
    
    def _negate(cls, name: str) -> type[DerivedFactor]:                         # noqa: N805
        """
        Return a dynamicly generated concrete subclass of `DerivedFactor`.
        
        The subclass' `REQUIRED_FEATURES` will be identical to that of `cls`'.
        The subclass' `compute` will be the negation of the result of
        `cls.compute`.
        """

        from .derived_factor import DerivedFactor
        
        required_features = cls.REQUIRED_FEATURES                               # type: ignore
        
        def _compute(
            cls_: type[DerivedFactor],
            x: xs.CrossSection,
            cache: FactorCache,
        ) -> xs.Array:
            return -cls.compute(x, cache)                                       # type: ignore

        return type(
            name,
            (DerivedFactor,),
            {
                "REQUIRED_FEATURES": required_features,
                "_compute": classmethod(_compute),
            },
        )
