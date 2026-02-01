from dataclasses import dataclass
from typing import TYPE_CHECKING, Type

from alpha_research_framework.features.window import Window

if TYPE_CHECKING:
    from alpha_research_framework.features.feature import Feature


@dataclass(frozen=True)
class FeatureSpec:
    """"""

    feature_cls: Type["Feature"]
    args: Window | tuple[Window,...]

    def __post_init__(self):
        if isinstance(self.args, Window):
            object.__setattr__(self, "args", (self.args,))

    def instantiate(self) -> "Feature":
        return self.feature_cls(*self.args)
