from typing import Any


class ClassVarValidator:
    """
    Base class for classes that require subclasses to define class variables
    with type and value constraints.
    """

    @classmethod
    def assert_class_var(
        cls,
        name: str,
        type: type,
        bad_values: set[Any] | None = None
    ) -> None:
        """
        Assert that a class variable exists, has the right type, and is not in
        bad_values.
        """

        if not hasattr(cls, name):
            raise AttributeError(
                f"{cls.__name__} must define class variable '{name}'"
            )
        value = getattr(cls, name)
        if not isinstance(value, type):
            raise TypeError(
                f"{cls.__name__}.{name} must be of type {type.__name__}, got "
                f"{value.__class__.__name__}"
            )
        if bad_values and value in bad_values:
            raise ValueError(
                f"{cls.__name__}.{name} cannot be one of {bad_values}"
            )

    @classmethod
    def assert_class_var_container(
        cls,
        name: str,
        container_type: type,
        element_type: type,
        bad_values: set[Any] | None = None
    ) -> None:
        """
        Assert that a class variable exists, is a container of the right type,
        all elements have the correct type, and no element is in bad_values.
        """

        cls.assert_class_var(name, container_type)

        value = getattr(cls, name)
        for i, element in enumerate(value):
            if not isinstance(element, element_type):
                raise TypeError(
                    f"{cls.__name__}.{name}[{i}] must be "
                    f"{element_type.__name__}, got {type(element).__name__}"
                )
            if bad_values and element in bad_values:
                raise ValueError(
                    f"{cls.__name__}.{name}[{i}] cannot be one of {bad_values}"
                )
