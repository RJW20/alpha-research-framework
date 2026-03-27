from abc import abstractmethod
from typing import Any, ClassVar

import alpha_research_framework.cross_section as xs
import alpha_research_framework.features as features
from alpha_research_framework.class_var_validator import ClassVarValidator
from alpha_research_framework.operator import Operator

from .factor_cache import FactorCache
from .factor_meta import FactorMeta


class Factor(Operator, ClassVarValidator, metaclass=FactorMeta):
    """
    Abstract base class for cross-sectional factors with automatic subclass
    validation.

    Should not be directly subclassed factors may be created by subclassing
    `PrimitiveFactor` or combining `Factor`s via the operators `+`, `-`, `*`,
    `/` resulting in a `DerivedFactor`.
    """

    # Features required to compute this factor
    REQUIRED_FEATURES: ClassVar[set[type[features.Feature]]]

    def __init_subclass__(cls, abstract: bool = False, **kwargs: Any) -> None:
        """
        If `abstract=False` asserts definition and type of `REQUIRED_FEATURES`.
        """

        super().__init_subclass__(**kwargs)
        if not abstract:
            cls.assert_class_var_container_of_subtype(
                name="REQUIRED_FEATURES",
                container_type=set,
                element_base_type=features.Feature,
            )

    @classmethod
    @abstractmethod
    def compute(cls, x: xs.CrossSection, cache: FactorCache) -> xs.Array:
        """
        Return an `xs.Array` containing raw cross-sectional value per stock.
        """
        ...
