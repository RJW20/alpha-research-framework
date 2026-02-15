from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from alpha_research_framework.dependent.dependent_error import DependentError

T = TypeVar('T')

class Dependent(ABC, Generic[T]):
    """
    Generic base class for objects that declare constant instance-level
    dependencies with automatic subclass validation and runtime dependency type
    enforcement.

    Any concrete subclass must define:
    - __dependency_type__: type = T
    - _init_dependencies(self) -> set[T]
    """

    __dependency_type__: type | None = None

    def __init__(self) -> None:
        """Validate type of and assign self._dependencies."""

        dependencies = self._init_dependencies()
        self._validate(dependencies)
        self._dependencies = frozenset(dependencies)

    def __init_subclass__(cls) -> None:
        """Validate definition and type of __dependency_type__."""

        super().__init_subclass__()

        if cls is Dependent:
            return
        
        if cls.__dependency_type__ is None:
            raise DependentError(
                f"{cls.__name__} must define __dependency_type__."
            )
        if not isinstance(cls.__dependency_type__, type):
            raise TypeError(
                f"{cls.__name__}.__dependency_type__ must be a type."
            )
    
    @property
    def dependencies(self) -> frozenset[T]:
        return self._dependencies

    @abstractmethod
    def _init_dependencies(self) -> set[T]:
        """Construct and return dependencies."""
        ...

    @classmethod
    def _validate(cls, dependencies: set[T]) -> None:
        """
        Enforce dependencies is of type set[cls.__dependency_type__].
        """

        if (
            not isinstance(dependencies, set) or
            not all(
                isinstance(d, cls.__dependency_type__) for d in dependencies
            )
        ):
            raise TypeError(
                f"{cls.__name__}._init_dependencies must have return type "
                f"set[{cls.__dependency_type__.__name__}]."
            )
