from typing import Any, ClassVar

from alpha_research_framework.class_var_validator import ClassVarValidator


class Observable(ClassVarValidator):
    """
    Absract base class for representing market observables with automatic
    subclass validation.
    """

    NAME: ClassVar[str]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Assert definition and type of `NAME`."""

        super().__init_subclass__(**kwargs)
        cls.assert_class_var("NAME", type=str)
