from abc import abstractmethod
from typing import Any

import alpha_research_framework.cross_section as xs
from alpha_research_framework.class_var_validator import ClassVarValidator
from alpha_research_framework.operator import Operator

from .cache import Cache
from .signal_meta import SignalMeta


class Signal(Operator, ClassVarValidator, metaclass=SignalMeta):
    """
    Abstract base class for cross-sectional signals.

    Should not be directly subclassed - signals may be created by subclassing
    `SeriesSignal` or combining/negating `Signals`s via the operators `+`, `-`,
    `*`, `/` resulting in a `CombinedSignal`/`NegatedSignal`.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

    @classmethod
    @abstractmethod
    def compute(cls, cross_section: xs.CrossSection, cache: Cache) -> xs.Array:
        """
        Return an `xs.Array` containing raw cross-sectional value per stock.

        Values are calculated from `cross_section` and/or already computed
        signals stored in `cache`.
        """
        ...
