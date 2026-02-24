import unittest
from typing import ClassVar

from alpha_research_framework.class_var_validator import ClassVarValidator


class TestAssertClassVar(unittest.TestCase):

    class NeedsClassVarValidated(ClassVarValidator):
        attr: ClassVar[str]
        def __init_subclass__(cls) -> None:
            cls.assert_class_var(name="attr", type=str, bad_values={""})

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
                TestAssertClassVar.NeedsClassVarValidated
            ):
                pass

    def test_container_type(self) -> None:
        """Verify type of container class var enforced."""

        with self.assertRaises(AttributeError):
            class NeedsClassVarValidatedSubClass(
                TestAssertClassVar.NeedsClassVarValidated
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


if __name__ == "__main__":
    unittest.main()
