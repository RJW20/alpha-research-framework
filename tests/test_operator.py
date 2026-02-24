import unittest

from alpha_research_framework.operator import Operator


class TestOperatorID(unittest.TestCase):

    class RegistryRoot(Operator, registry_root=True, abstract=True):
        pass

    def test_duplicate(self) -> None:
        """Verify 2 operators with the same root cannot share `ID`."""

        class Operator1(TestOperatorID.RegistryRoot):
            ID = "op"

        with self.assertRaises(ValueError):
            class Operator2(TestOperatorID.RegistryRoot):
                ID = "op"

    def test_from_id(self) -> None:
        """Verify operator returned by `id` matches."""

        class DummyOperator(TestOperatorID.RegistryRoot):
            ID = "dummy_op"
        self.assertIs(
            TestOperatorID.RegistryRoot.from_id("dummy_op"),
            DummyOperator
        )

    def test_from_invalid_id(self) -> None:
        """Verify an error is raised when `id` is fictional."""

        with self.assertRaises(ValueError):
            TestOperatorID.RegistryRoot.from_id("abc")


class TestOperatorRegistryRoot(unittest.TestCase):

    def test_mutually_exclusive_roots(self) -> None:
        """
        Verify a subclass of one root is not registered in another unrelated
        root.
        """

        class Root1(Operator, registry_root=True, abstract=True):
            pass

        class Root2(Operator, registry_root=True, abstract=True):
            pass

        class SubClassOfRoot1(Root1):
            ID = "subclass_of_root_1"

        with self.assertRaises(ValueError):
            Root2.from_id("subclass_of_root_1")

    def test_no_root_in_inheritance_chain(self) -> None:
        """
        Verify a subclass that is not a descendent of a root cannot be
        non-abstract.
        """

        with self.assertRaises(TypeError):
            class RootlessSubClass(Operator):
                pass

    def test_multiple_roots_in_inheritance_chain(self) -> None:
        """
        Verify a subclass that is a descendent of multiple unique roots is
        registered in both.
        """

        class Root1(Operator, registry_root=True, abstract=True):
            pass

        class Root2(Operator, registry_root=True, abstract=True):
            pass

        class SubClassOfRoot1And2(Root1, Root2):
            ID = "subclass_of_root_1_2"

        Root1.from_id("subclass_of_root_1_2")
        Root2.from_id("subclass_of_root_1_2")

    def test_repeated_root_in_inheritance_chain(self) -> None:
        """
        Verify a subclass that is a double descendent of a root is only
        registered once.
        """

        class Root(Operator, registry_root=True, abstract=True):
            pass

        class SubClassOfRoot(Root, abstract=True):
            pass

        class DoubleSubClassOfRoot(SubClassOfRoot, Root):
            ID = "double_sub_class"


if __name__ == "__main__":
    unittest.main()
