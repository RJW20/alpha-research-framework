import unittest

import alpha_research_framework.signals as signals
from alpha_research_framework.alphas import Alpha
from tests.utils import RegistryIsolatedTestCase


class TestAlphaCategory(RegistryIsolatedTestCase):
    REGISTRY_OWNER = Alpha

    def test_class_var_assertion(self) -> None:
        """
        Verify definition and type of `CATEGORY` asserted when subclassing.
        """

        with self.assertRaises(AttributeError):
            class NoCategory(Alpha):
                ID = "no_category"

        with self.assertRaises(TypeError):
            class IncompatibleCategory(Alpha):
                ID = "incompatible_tag"
                CATEGORY = 1


class TestAlphaSignal(RegistryIsolatedTestCase):
    REGISTRY_OWNER = Alpha

    def test_class_var_assertion(self) -> None:
        """Verify definition and type of `SIGNAL` asserted when subclassing."""

        with self.assertRaises(AttributeError):
            class NoSignal(Alpha):
                ID = "no_signal"
                CATEGORY = "testing_signal"

        with self.assertRaises(TypeError):
            class IncompatibleSignal(Alpha):
                ID = "incompatible_signal"
                CATEGORY = "testing_signal"
                SIGNAL = str


class TestAlphaHorizons(RegistryIsolatedTestCase):
    REGISTRY_OWNER = Alpha

    def test_class_var_assertion(self) -> None:
        """
        Verify definition and type of `HORIZONS` asserted when subclassing.
        """

        class Dummy(signals.Signal):
            pass

        with self.assertRaises(AttributeError):
            class NoHorizons(Alpha):
                ID = "no_horizons"
                CATEGORY = "testing_horizons"
                SIGNAL = Dummy

        with self.assertRaises(TypeError):
            class IncompatibleHorizonsContainer(Alpha):
                ID = "incompatible_horizons_container"
                CATEGORY = "testing_horizons"
                SIGNAL = Dummy
                HORIZONS = list()

        with self.assertRaises(TypeError):
            class IncompatibleHorizonsElement(Alpha):
                ID = "incompatible_horizons_element"
                CATEGORY = "testing_horizons"
                SIGNAL = Dummy
                HORIZONS = {1}


if __name__ == "__main__":
    unittest.main()
