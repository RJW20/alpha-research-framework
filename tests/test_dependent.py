import unittest

from alpha_research_framework.dependent import Dependent
from alpha_research_framework.dependent.dependent_error import DependentError


class TestDependentSubClassValidation(unittest.TestCase):

    def test_dependency_type(self) -> None:
        """
        Verify definition and type of __dependency_type__ validated when
        subclassing.
        """

        with self.assertRaises(DependentError):
            class Dummy(Dependent[int]):
                pass

        with self.assertRaises(TypeError):
            class Dummy(Dependent[int]):
                __dependency_type__ = 1


class TestDependentTypeEnforcement(unittest.TestCase):

    def test_validate(self) -> None:
        """Verify type of dependencies enforced in __init__ via _validate."""

        with self.assertRaises(TypeError):
            class Dummy(Dependent[int]):
                __dependency_type__ = int
                def _init_dependencies(self) -> set[int]:
                    return [1, 2 ,3]
            dummy = Dummy()

        with self.assertRaises(TypeError):
            class Dummy(Dependent[int]):
                __dependency_type__ = int
                def _init_dependencies(self) -> set[int]:
                    return {"a", 123}
            dummy = Dummy()


if __name__ == "__main__":
    unittest.main()
