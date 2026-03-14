from __future__ import annotations

from abc import ABCMeta
from typing import TYPE_CHECKING, Callable

import alpha_research_framework.market_data as md
from alpha_research_framework.universe import CrossSection

if TYPE_CHECKING:
    from alpha_research_framework.alphas.alpha.alpha import Alpha


class AlphaMeta(ABCMeta):
    """AlphaMeta is a metaclass that provides utility for combining alphas."""

    def _compose(
        cls: type[Alpha],
        other: type[Alpha],
        fn: Callable[[md.Array, md.Array], md.Array],
        name: str,
    ) -> type[Alpha]:
        """
        Return a dynamicly generated abstract subclass of `Alpha`.
        
        The subclass' `DEPENDENCIES` will be the union of cls' and other's
        `DEPENDENCIES`.
        The subclass' `compute` will be the application of `fn` to the results
        of `cls.compute` and `other.compute`.
        """

        from alpha_research_framework.alphas.alpha.alpha import Alpha

        if not issubclass(other, Alpha):
            raise TypeError(
                f"Unable to build new alpha: {other.__name__} must be a "
                "subclass of Alpha"
            )
        
        def compute(cls_: type[Alpha], x: CrossSection) -> md.Array:
            return fn(cls.compute(x), other.compute(x))

        ComposedAlpha = type(
            name,
            (Alpha,),
            {
                "DEPENDENCIES": cls.DEPENDENCIES | other.DEPENDENCIES,
                "compute": classmethod(compute),
            },
            abstract=True,
        )
    
        return ComposedAlpha

    def __add__(cls: type[Alpha], other: type[Alpha]) -> type[Alpha]:
        return cls._compose(
            other,
            lambda a, b: a + b,
            f"({cls.__name__}Plus{other.__name__})",
        )

    def __sub__(cls: type[Alpha], other: type[Alpha]) -> type[Alpha]:
        return cls._compose(
            other,
            lambda a, b: a - b,
            f"({cls.__name__}Minus{other.__name__})",
        )

    def __mul__(cls: type[Alpha], other: type[Alpha]) -> type[Alpha]:
        return cls._compose(
            other,
            lambda a, b: a * b,
            f"({cls.__name__}Times{other.__name__})",
        )

    def __truediv__(cls: type[Alpha], other: type[Alpha]) -> type[Alpha]:
        return cls._compose(
            other,
            lambda a, b: a / b,
            f"({cls.__name__}Divide{other.__name__})"
        )
