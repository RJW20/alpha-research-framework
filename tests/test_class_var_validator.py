import unittest
from typing import Any, Callable, ClassVar

from alpha_research_framework.class_var_validator import ClassVarValidator


class TestAssertClassVar(unittest.TestCase):

    class NeedsClassVarValidated(ClassVarValidator):
        attr: ClassVar[str]
        def __init_subclass__(cls) -> None:
            cls.assert_class_var("attr", type=str, bad_values={""})

    def test_definition(self) -> None:
        """Verify definition of class var enforced."""

        with self.assertRaises(AttributeError):
            class NeedsClassVarValidatedSubClass(
                TestAssertClassVar.NeedsClassVarValidated
            ):
                pass

    def test_type(self) -> None:
        """Verify type of class var enforced."""

        with self.assertRaises(TypeError):
            class NeedsClassVarValidatedSubClass(
                TestAssertClassVar.NeedsClassVarValidated
            ):
                attr = 1

    def test_bad_values(self) -> None:
        """Verify no bad values for class var enforced."""

        with self.assertRaises(ValueError):
            class NeedsClassVarValidatedSubClass(
                TestAssertClassVar.NeedsClassVarValidated
            ):
                attr = ""


class TestAssertClassVarSubType(unittest.TestCase):

    class NeedsClassVarValidated(ClassVarValidator):
        attr: ClassVar[type[str]]
        def __init_subclass__(cls) -> None:
            cls.assert_class_var_subtype("attr", base_type=str)

    def test_definition(self) -> None:
        """Verify definition of class var enforced."""

        with self.assertRaises(AttributeError):
            class NeedsClassVarValidatedSubClass(
                TestAssertClassVarSubType.NeedsClassVarValidated
            ):
                pass

    def test_type(self) -> None:
        """Verify class var is a type enforced."""

        with self.assertRaises(TypeError):
            class NeedsClassVarValidatedSubClass(
                    TestAssertClassVarSubType.NeedsClassVarValidated
                ):
                    attr = "a"

    def test_base_type(self) -> None:
        """Verify base type of class var enforced."""

        with self.assertRaises(TypeError):
            class NeedsClassVarValidatedSubClass(
                TestAssertClassVarSubType.NeedsClassVarValidated
            ):
                attr = int


class TestAssertClassVarContainer(unittest.TestCase):

    class NeedsClassVarContainerValidated(ClassVarValidator):
        attrs: ClassVar[set[str]]
        def __init_subclass__(cls) -> None:
            cls.assert_class_var_container(
                name="attrs",
                container_type=set,
                element_type=str,
                bad_values={""},
            )

    def test_definition(self) -> None:
        """Verify definition of container class var enforced."""

        with self.assertRaises(AttributeError):
            class NeedsClassVarValidatedSubClass(
                TestAssertClassVarContainer.NeedsClassVarContainerValidated
            ):
                pass

    def test_container_type(self) -> None:
        """Verify type of container class var enforced."""

        with self.assertRaises(TypeError):
            class NeedsClassVarValidatedSubClass(
                TestAssertClassVarContainer.NeedsClassVarContainerValidated
            ):
                attrs = ["attr1", "attr2"]

    def test_element_type(self) -> None:
        """Verify type of container class var elements enforced."""

        with self.assertRaises(TypeError):
            class NeedsClassVarContainerValidatedSubClass(
                TestAssertClassVarContainer.NeedsClassVarContainerValidated
            ):
                attrs = {1}

    def test_bad_values(self) -> None:
        """Verify no bad values in container class var enforced."""

        with self.assertRaises(ValueError):
            class NeedsClassVarContainerValidatedSubClass(
                TestAssertClassVarContainer.NeedsClassVarContainerValidated
            ):
                attrs = {""}


class TestAssertClassVarContainerOfSubType(unittest.TestCase):

    class NeedsClassVarContainerValidated(ClassVarValidator):
        attrs: ClassVar[set[type[str]]]
        def __init_subclass__(cls) -> None:
            cls.assert_class_var_container_of_subtype(
                name="attrs",
                container_type=set,
                element_base_type=str,
            )

    def test_definition(self) -> None:
        """Verify definition of container class var enforced."""

        with self.assertRaises(AttributeError):
            class NeedsClassVarValidatedSubClass(
                TestAssertClassVarContainerOfSubType
                .NeedsClassVarContainerValidated
            ):
                pass

    def test_container_type(self) -> None:
        """Verify type of container class var enforced."""

        with self.assertRaises(TypeError):
            class NeedsClassVarValidatedSubClass(
                TestAssertClassVarContainerOfSubType
                .NeedsClassVarContainerValidated
            ):
                attrs = [str, str]

    def test_element_type(self) -> None:
        """Verify container class var elements are types enforced."""

        with self.assertRaises(TypeError):
            class NeedsClassVarContainerValidatedSubClass(
                TestAssertClassVarContainerOfSubType
                .NeedsClassVarContainerValidated
            ):
                attrs = {"a"}

    def test_element_base_type(self) -> None:
        """Verify base type of container class var elements enforced."""

        with self.assertRaises(TypeError):
            class NeedsClassVarContainerValidatedSubClass(
                TestAssertClassVarContainerOfSubType
                .NeedsClassVarContainerValidated
            ):
                attrs = {int}


class TestAssertClassVarFunction(unittest.TestCase):

    class NeedsClassVarFunctionValidated(ClassVarValidator):
        attr: ClassVar[Callable[[Any, Any], Any]]
        def __init_subclass__(cls) -> None:
            def expected_attr(x: Any, y: Any) -> Any:
                ...
            cls.assert_class_var_function("attr", prototype=expected_attr)

    def test_definition(self) -> None:
        """Verify definition of function class var enforced."""

        with self.assertRaises(AttributeError):
            class NeedsClassVarValidatedSubClass(
                TestAssertClassVarFunction.NeedsClassVarFunctionValidated
            ):
                pass

    def test_arguments(self) -> None:
        """Verify number of arguments of function class var enforced."""

        with self.assertRaises(TypeError):
            class NeedsClassVarValidatedSubClass(
                TestAssertClassVarFunction.NeedsClassVarFunctionValidated
            ):
                attr = lambda x: x*2


if __name__ == "__main__":
    unittest.main()
