import unittest

import numpy as np

import alpha_research_framework.features as features
import alpha_research_framework.market_data as md
from alpha_research_framework import Window
from alpha_research_framework.alphas import Alpha
from alpha_research_framework.alphas.returns_based import ReturnsBased
from tests.utils import RegistryIsolatedTestCase


class TestReturnsBasedLookback(RegistryIsolatedTestCase):
    REGISTRY_OWNER = Alpha

    def test_class_var_assertion(self) -> None:
        """
        Verify definition and type of `LOOKBACK` asserted when subclassing.
        """

        with self.assertRaises(AttributeError):
            class NoLookback(ReturnsBased):
                ID = "no_lookback"
                CATEGORY = "testing_lookback"

        with self.assertRaises(TypeError):
            class IncompatibleLookback(ReturnsBased):
                ID = "incompatible_lookback"
                CATEGORY = "testing_lookback"
                LOOKBACK = 1


class TestReturnsBasedSkip(RegistryIsolatedTestCase):
    REGISTRY_OWNER = Alpha

    def test_class_var_assertion(self) -> None:
        """Verify type and value of `SKIP` asserted when subclassing."""

        with self.assertRaises(TypeError):
            class IncompatibleSkip(ReturnsBased):
                ID = "incompatible_skip"
                CATEGORY = "testing_skip"
                LOOKBACK = Window.DAY
                SKIP = 1

        with self.assertRaises(ValueError):
            class LargerSkip(ReturnsBased):
                ID = "larger_skip"
                CATEGORY = "testing_skip"
                LOOKBACK = Window.DAY
                SKIP = Window.DAY


class TestReturnsBasedDependencies(RegistryIsolatedTestCase):
    REGISTRY_OWNER = Alpha

    def test_no_skip(self) -> None:
        """Verify only `_RETURNS_LOOKBACK` dependency created."""

        class NoSkip(ReturnsBased):
            ID = "no_skip"
            CATEGORY = "testing_dependencies"
            LOOKBACK = Window.DAY
            HORIZONS = set()

        self.assertIs(NoSkip._RETURNS_LOOKBACK, features.DailyReturns)
        self.assertIsNone(NoSkip._RETURNS_SKIP)
        self.assertEqual(len(NoSkip.DEPENDENCIES), 1)

    def test_with_skip(self) -> None:
        """
        Verify both `_RETURNS_LOOKBACK` and `_RETURNS_SKIP` dependencies
        created.
        """

        class WithSkip(ReturnsBased):
            ID = "with_skip"
            CATEGORY = "testing_dependencies"
            LOOKBACK = Window.WEEK
            SKIP = Window.DAY
            HORIZONS = set()

        self.assertIs(WithSkip._RETURNS_LOOKBACK, features.WeeklyReturns)
        self.assertIs(WithSkip._RETURNS_SKIP, features.DailyReturns)
        self.assertEqual(len(WithSkip.DEPENDENCIES), 2)


class TestReturnsBasedCompute(RegistryIsolatedTestCase):
    REGISTRY_OWNER = Alpha

    T = 5000
    N = 500

    def test_compute(self) -> None:
        """
        Verify `compute` returns the returns over `LOOKBACK` or between
        `LOOKBACK` and `SKIP`.
        """

        rng = np.random.default_rng(0)
        
        windows_return_pairs: list[tuple[Window, type[features.Returns]]] = [
            (Window.DAY, features.DailyReturns),
            (Window.WEEK, features.WeeklyReturns),
            (Window.MONTH, features.MonthlyReturns),
            (Window.QUARTER, features.QuarterlyReturns),
            (Window.HALF_YEAR, features.HalfYearlyReturns),
            (Window.YEAR, features.YearlyReturns),
        ]
        for lookback, lookback_returns in windows_return_pairs:

            class LookbackOnly(ReturnsBased):
                ID = f"lookback_{lookback.value}_only"
                CATEGORY = "testing_compute"
                LOOKBACK = lookback
                HORIZONS = set()

            returns_over_lookback = rng.uniform(
                0,
                10,
                size=(TestReturnsBasedCompute.T, TestReturnsBasedCompute.N),
            ).astype(md.Scalar)
            nan_mask = rng.uniform(size=returns_over_lookback.shape) < 0.1
            returns_over_lookback[nan_mask] = np.nan
            x = {lookback_returns.ID: returns_over_lookback}
            np.testing.assert_array_equal(
                LookbackOnly.compute(x),
                returns_over_lookback,
            )
            
            for skip, skip_returns in windows_return_pairs:
                if not skip < lookback:
                    continue

                class LookbackAndSkip(ReturnsBased):
                    ID = f"lookback_{lookback.value}_skip_{skip.value}"
                    CATEGORY = "testing_compute"
                    LOOKBACK = lookback
                    SKIP = skip
                    HORIZONS = set()

                returns_over_skip = rng.uniform(
                    0,
                    10,
                    size=(TestReturnsBasedCompute.T, TestReturnsBasedCompute.N),
                ).astype(md.Scalar)
                nan_mask = rng.uniform(size=returns_over_skip.shape) < 0.1
                returns_over_skip[nan_mask] = np.nan
                x = {
                    lookback_returns.ID: returns_over_lookback,
                    skip_returns.ID: returns_over_skip,
                }
                np.testing.assert_array_equal(
                    LookbackAndSkip.compute(x),
                    returns_over_lookback - returns_over_skip,
                )


if __name__ == "__main__":
    unittest.main()
