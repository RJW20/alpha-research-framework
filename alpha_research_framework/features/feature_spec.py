from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Callable, Type

from alpha_research_framework.market_data_view import (
    MarketArray,
    MarketDataView,
)
from alpha_research_framework.window import Window

if TYPE_CHECKING:
    from alpha_research_framework.features.feature import Feature
    from alpha_research_framework.features.features import Features


@dataclass(frozen=True)
class FeatureSpec:
    """
    Immutable specification of a feature class and its window arguments.

    Acts as a lightweight, hashable descriptor that lazily instantiates
    the underlying feature when needed.
    """

    cls: Type["Feature"]
    windows: tuple[Window,...]

    def __init__(self, cls: Type["Feature"], *windows: Window) -> None:
        object.__setattr__(self, "cls", cls)
        object.__setattr__(self, "windows", windows)
    
    @property
    def name(self) -> str:
        return self._underlying.name
    
    @property
    def tag(self) -> Type["Feature.Tag"]:
        return self._underlying.TAG
    
    @property
    def dependencies(self) -> frozenset["FeatureSpec"]:
        return self._underlying.dependencies
    
    @property
    def compute(
        self
    ) -> Callable[[MarketDataView, "Features", MarketArray], None]:
        return self._underlying.compute
        
    @cached_property
    def _underlying(self) -> "Feature":
        return self.cls(*self.windows)
