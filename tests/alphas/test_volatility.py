import unittest

from alpha_research_framework.alphas.alpha_error import AlphaError
from alpha_research_framework.alphas.volatility import Volatility


class TestReturnsBasedSubclassValidation(unittest.TestCase):

    def test_abstract(self) -> None:

        class Dummy(Volatility):
            __abstract__ = True

    def test_lookback(self) -> None:

        with self.assertRaises(AlphaError):
            class Dummy(Volatility):
                NAME = "dummy"
                CATEGORY = "dummy"
                HORIZONS = set()

        with self.assertRaises(TypeError):
            class Dummy(Volatility):
                NAME = "dummy"
                CATEGORY = "dummy"
                HORIZONS = set()
                LOOKBACK = 1


if __name__ == "__main__":
    unittest.main()
