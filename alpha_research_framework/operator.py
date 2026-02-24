from __future__ import annotations

from abc import ABC
from typing import Any, ClassVar

from alpha_research_framework.class_var_validator import ClassVarValidator


class Operator(ClassVarValidator, ABC):
    """
    Abstract base class for registry-aware operators.

    Responsibilities:
    - Enforce that concrete subclasses define a unique string `ID`.
    - Maintain a per-registry-root-class `__registry__` of concrete subclasses.
    - Provide a classmethod `from_id` for lookup.

    When subclassing:
    - Passing `registry_root=True` initialises a new `__registry__` for the
    subclass (and it's subclasses).
    - Passing `abstract=True` prevents the requirement of `ID` and doesn't
    register the subclass.
    """

    # Only meaningful for registry roots
    __registry__: ClassVar[dict[str, type[Operator]]]

    # Each concrete subclass must define a unique string ID
    ID: ClassVar[str]

    def __init_subclass__(
        cls,
        *,
        registry_root: bool = False,
        abstract: bool = False,
        **kwargs: Any
    ) -> None:
        """
        Initialise a new subclass.

        If `registry_root=True` initialises a new `__registry__`.
        If `abstract=False` enforces definition, type, value and uniqueness of
        `ID` and adds subclass to `__registry__`.
        """

        super().__init_subclass__(**kwargs)

        if registry_root:
            cls.__registry__ = dict()

        if abstract:
            return
        
        root_registries = [
            (base, base.__registry__)
            for base in set(cls.__mro__[1:])
            if "__registry__" in base.__dict__
        ]
        if not root_registries:
            raise TypeError(
                f"{cls.__name__} is concrete but does not have a registry. "
                "Ensure a registry root exists in the inheritance chain."
            )
        
        cls.assert_class_var(name="ID", type=str, bad_values={""})
        for base, registry in root_registries:
            if cls.ID in registry:
                raise ValueError(
                    f"Duplicate ID '{cls.ID}' found in registry for "
                    f"{base.__name__}"
                )
            registry[cls.ID] = cls

    @classmethod
    def from_id(cls, id: str) -> type[Operator]:
        """Return the operator with ID = id."""
        if id not in cls.__registry__:
            raise ValueError(
                f"ID '{id}' not found in registry for {cls.__name__}"
            )
        return cls.__registry__[id]
