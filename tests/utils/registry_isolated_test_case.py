import unittest
from typing import ClassVar

from alpha_research_framework.class_var_validator import ClassVarValidator
from alpha_research_framework.registrable import Registrable


class RegistryIsolatedTestCase(unittest.TestCase, ClassVarValidator):
    """
    Class for ensuring a `registry_root=True` registrable's `__registry__`
    class variable is not permanently mutated during tests.

    All subclasses defined in one test scope must still have unique names.
    """

    REGISTRY_OWNER: ClassVar[type[Registrable]]

    def __init_subclass__(cls) -> None:
        cls.assert_class_var(name="REGISTRY_OWNER", type=type)

    def setUp(self):
        super().setUp()
        self._old_registry = self.REGISTRY_OWNER.__registry__.copy()
        self.REGISTRY_OWNER.__registry__.clear()

    def tearDown(self):
        self.REGISTRY_OWNER.__registry__.clear()
        self.REGISTRY_OWNER.__registry__.update(self._old_registry)
        super().tearDown()
