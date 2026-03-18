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
        *,
        type: type,
        bad_values: set[Any] | None = None,
    ) -> None:
        """
        Assert that a class variable exists, is of type `type` and is not in
        `bad_values`.
        """

        if not hasattr(cls, name):
            raise AttributeError(
                f"{cls.__name__} must define class variable '{name}'"
            )
        class_var = getattr(cls, name)
        if not isinstance(class_var, type):
            raise TypeError(
                f"{cls.__name__}.{name} must be of type {type.__name__}, got "
                f"{class_var.__class__.__name__}"
            )
        if bad_values and class_var in bad_values:
            raise ValueError(
                f"{cls.__name__}.{name} cannot be one of {bad_values}"
            )
        
    @classmethod
    def assert_class_var_subtype(cls, name: str, *, base_type: type) -> None:
        """
        Assert that a class variable exists, is of type `type` and is a subclass
        of `base_type`.
        """

        cls.assert_class_var(name, type=type)

        class_var = getattr(cls, name)
        if not issubclass(class_var, base_type):
            raise TypeError(
                f"{cls.__name__}.{name} must be a subclass of type "
                f"{base_type.__name__}"
            )

    @classmethod
    def assert_class_var_container(
        cls,
        name: str,
        *,
        container_type: type,
        element_type: type,
        bad_values: set[Any] | None = None,
    ) -> None:
        """
        Assert that a class variable exists, is of type `container_type`,
        contains only `element_type`, and contains no element in `bad_values`.
        """

        cls.assert_class_var(name, type=container_type)

        class_var = getattr(cls, name)
        for element in class_var:
            if not isinstance(element, element_type):
                raise TypeError(
                    f"{cls.__name__}.{name} must exclusively contain type "
                    f"{element_type.__name__}, got {type(element).__name__}"
                )
            if bad_values and element in bad_values:
                raise ValueError(
                    f"{cls.__name__}.{name} cannot contain one of {bad_values}"
                )

    @classmethod
    def assert_class_var_container_of_subtype(
        cls,
        name: str,
        *,
        container_type: type,
        element_base_type: type,
    ) -> None:
        """
        Assert that a class variable exists, is of type `container_type` and
        contains elements of type `type` which are all subclasses of
        `element_base_type`.
        """

        cls.assert_class_var_container(
            name,
            container_type=container_type,
            element_type=type,
        )

        class_var = getattr(cls, name)
        for element in class_var:
            if not issubclass(element, element_base_type):
                raise TypeError(
                    f"{cls.__name__}.{name} must exclusively contain "
                    f"subclasses of type {element_base_type.__name__}"
                )
