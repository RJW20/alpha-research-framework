import unittest

import numpy as np

from alpha_research_framework import Window
from alpha_research_framework.alphas.alpha_error import AlphaError
from alpha_research_framework.alphas.returns_based import ReturnsBased
from alpha_research_framework.features import Returns
from alpha_research_framework.universe import CrossSection


class TestReturnsBasedSubclassValidation(unittest.TestCase):

    def test_abstract(self) -> None:
        """Verify subclass is not validated when __abstract__ is set."""

        class Dummy(ReturnsBased):
            __abstract__ = True

    def test_lookback(self) -> None:
        """
        Verify definition and type of LOOKBACK validated when subclassing.
        """

        with self.assertRaises(AlphaError):
            class Dummy(ReturnsBased):
                NAME = "dummy"
                CATEGORY = "dummy"
                HORIZONS = set()

        with self.assertRaises(TypeError):
            class Dummy(ReturnsBased):
                NAME = "dummy"
                CATEGORY = "dummy"
                HORIZONS = set()
                LOOKBACK = 1

    def test_skip(self) -> None:
        """Verify type and value of SKIP validated when subclassing."""

        with self.assertRaises(TypeError):
            class Dummy(ReturnsBased):
                NAME = "dummy"
                CATEGORY = "dummy"
                HORIZONS = set()
                LOOKBACK = Window.DAY
                SKIP = 1

        with self.assertRaises(ValueError):
            class Dummy(ReturnsBased):
                NAME = "dummy"
                CATEGORY = "dummy"
                HORIZONS = set()
                LOOKBACK = Window.DAY
                SKIP = Window.DAY


class TestReturnsBasedDependencies(unittest.TestCase):

    def test_init(self) -> None:
        """
        Verify the number of dependencies created depending on the definition
        of SKIP.
        """

        class Dummy(ReturnsBased):
            __abstract__ = True
            LOOKBACK = Window.WEEK

        dummy = Dummy()
        self.assertEqual(len(dummy._init_dependencies()), 1)

        Dummy.SKIP = Window.DAY
        self.assertEqual(len(dummy._init_dependencies()), 2)


class TestReturnsBasedCompute(unittest.TestCase):

    def test_compute(self) -> None:
        """
        Verify compute returns the returns over LOOKBACK or between LOOKBACK
        and SKIP.
        """

        rng = np.random.default_rng(0)

        for lookback in Window:

            x = CrossSection()
            returns_lookback = Returns(lookback)
            x[returns_lookback.name] = \
                rng.uniform(0, 10, 100).astype(np.float32)
            
            class DummyNoSkip(ReturnsBased):
                NAME = "dummy"
                CATEGORY = "dummy"
                HORIZONS = set()
                LOOKBACK = lookback
            dummy = DummyNoSkip()

            np.testing.assert_array_equal(
                    dummy.compute(x),
                    x[returns_lookback.name]
                )
            
            for skip in [w for w in Window if w < lookback]:

                returns_skip = Returns(skip)
                x[returns_skip.name] = \
                    rng.uniform(0, 10, 100).astype(np.float32)

                class Dummy(ReturnsBased):
                    NAME = "dummy"
                    CATEGORY = "dummy"
                    HORIZONS = set()
                    LOOKBACK = lookback
                    SKIP = skip
                dummy = Dummy()

                np.testing.assert_array_equal(
                    dummy.compute(x),
                    x[returns_lookback.name] - x[returns_skip.name]
                )


if __name__ == "__main__":
    unittest.main()
