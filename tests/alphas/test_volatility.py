import unittest

from alpha_research_framework.alphas import Alpha
from alpha_research_framework.alphas.alpha_error import AlphaError
from alpha_research_framework.alphas.volatility import Volatility
from tests.utils import RegistryIsolatedTestCase


class TestVolatilitySubClassValidation(RegistryIsolatedTestCase):
    REGISTRY_OWNER = Alpha

    def test_abstract(self) -> None:
        """Verify subclass is not validated when __abstract__ is set."""

        class Dummy(Volatility):
            __abstract__ = True

    def test_lookback(self) -> None:
        """
        Verify definition and type of LOOKBACK validated when subclassing.
        """

        with self.assertRaises(AlphaError):
            class Dummy1(Volatility):
                NAME = "dummy_1"
                CATEGORY = "dummy"
                HORIZONS = set()

        with self.assertRaises(TypeError):
            class Dummy2(Volatility):
                NAME = "dummy_2"
                CATEGORY = "dummy"
                HORIZONS = set()
                LOOKBACK = 1


if __name__ == "__main__":
    unittest.main()
