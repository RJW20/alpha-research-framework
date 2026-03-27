from abc import abstractmethod
from typing import Any

import alpha_research_framework.market_data as md
from alpha_research_framework.operator import Operator

from .transform_meta import TransformMeta


class Transform(Operator, metaclass=TransformMeta):
    """Abstract base class for market feature transforms."""

    @classmethod
    @abstractmethod
    def compute(cls, arr: md.Array, **kwargs: Any) -> None:
        """Apply a transformation to arr in-place."""
        ...
