import unittest
from typing import Protocol


class HasRegistry(Protocol):
    __registry__: dict[str, type]


class RegistryIsolatedTestCase(unittest.TestCase):
    """
    Class for ensuring a class' __registry__ attribute is not mutated duing
    tests.
    """

    REGISTRY_OWNER: type[HasRegistry] | None = None

    def setUp(self):
        super().setUp()
        self._old_registry = self.REGISTRY_OWNER.__registry__.copy()
        self.REGISTRY_OWNER.__registry__.clear()

    def tearDown(self):
        self.REGISTRY_OWNER.__registry__.clear()
        self.REGISTRY_OWNER.__registry__.update(self._old_registry)
        super().tearDown()
