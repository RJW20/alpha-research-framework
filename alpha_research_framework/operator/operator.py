from abc import ABC, abstractmethod
from typing import Any

from .operator_meta import OperatorMeta


class Operator(ABC, metaclass=OperatorMeta):
    """
    Abstract base class for stateless operators.
    
    Defines the interface.
    """

    @classmethod
    @abstractmethod
    def compute(cls, *args: Any, **kwargs: Any) -> Any:
        ...
