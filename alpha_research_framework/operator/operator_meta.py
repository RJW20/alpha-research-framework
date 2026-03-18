from abc import ABCMeta
from typing import Any


class OperatorMeta(ABCMeta):
    """
    Metaclass for stateless operators.
    
    Implements forwarding from `Operator(...)` to `Operator.compute(...)` (and
    as a result prevents instantiation).
    """

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        return cls.compute(*args, **kwargs)                                     # type: ignore[attr-defined]
