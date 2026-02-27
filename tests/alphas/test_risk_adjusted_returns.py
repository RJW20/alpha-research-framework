import unittest

import numpy as np

import alpha_research_framework.features as features
import alpha_research_framework.market_data as md
from alpha_research_framework import Window
from alpha_research_framework.alphas import Alpha
from alpha_research_framework.alphas.risk_adjusted_returns import (
    RiskAdjustedReturns,
)
from tests.utils import RegistryIsolatedTestCase


class TestRiskAdjustedReturnsLookback(RegistryIsolatedTestCase):
    REGISTRY_OWNER = Alpha

    def test_class_var_assertion(self) -> None:
        """
        Verify definition and type of `RETURNS_LOOKBACK` asserted when
        subclassing.
        """

        with self.assertRaises(AttributeError):
            class NoReturnsLookback(RiskAdjustedReturns):
                ID = "no_returns_lookback"
                CATEGORY = "testing_returns_lookback"

        with self.assertRaises(TypeError):
            class IncompatibleReturnsLookback(RiskAdjustedReturns):
                ID = "incompatible_returns_lookback"
                CATEGORY = "testing_returns_lookback"
                RETURNS_LOOKBACK = 1


class TestRiskAdjustedReturnsSkip(RegistryIsolatedTestCase):
    REGISTRY_OWNER = Alpha

    def test_class_var_assertion(self) -> None:
        """
        Verify definition and type of `RETURNS_SKIP` asserted when subclassing.
        """

        with self.assertRaises(TypeError):
            class NoReturnsSkip(RiskAdjustedReturns):
                ID = "no_returns_skip"
                CATEGORY = "testing_returns_skip"
                RETURNS_LOOKBACK = Window.DAY
                RETURNS_SKIP = 1

        with self.assertRaises(ValueError):
            class LargerReturnsSkip(RiskAdjustedReturns):
                ID = "incompatible_returns_skip"
                CATEGORY = "testing_returns_skip"
                RETURNS_LOOKBACK = Window.DAY
                RETURNS_SKIP = Window.DAY


class TestRiskAdjustedVolatilityLookback(RegistryIsolatedTestCase):
    REGISTRY_OWNER = Alpha

    def test_class_var_assertion(self) -> None:
        """
        Verify definition and type of `VOLATILITY_LOOKBACK` asserted when
        subclassing.
        """

        with self.assertRaises(AttributeError):
            class NoVolatilityLookback(RiskAdjustedReturns):
                ID = "no_volatility_lookback"
                CATEGORY = "testing_volatility_lookback"
                RETURNS_LOOKBACK = Window.DAY

        with self.assertRaises(TypeError):
            class IncompatibleVolatilityLookback(RiskAdjustedReturns):
                ID = "incompatible_volatility_lookback"
                CATEGORY = "testing_volatility_lookback"
                RETURNS_LOOKBACK = Window.DAY
                VOLATILITY_LOOKBACK = 1


class TestRiskAdjustedDependencies(RegistryIsolatedTestCase):
    REGISTRY_OWNER = Alpha

    def test_no_returns_skip(self) -> None:
        """
        Verify only `_RETURNS_LOOKBACK` and `_VOLATILITY_LOOKBACK` dependencies
        created.
        """

        class NoReturnsSkip(RiskAdjustedReturns):
            ID = "no_returns_skip"
            CATEGORY = "testing_dependencies"
            RETURNS_LOOKBACK = Window.DAY
            VOLATILITY_LOOKBACK = Window.WEEK
            HORIZONS = set()

        self.assertIs(NoReturnsSkip._RETURNS_LOOKBACK, features.DailyReturns)
        self.assertIsNone(NoReturnsSkip._RETURNS_SKIP)
        self.assertIs(
            NoReturnsSkip._VOLATILITY_LOOKBACK,
            features.WeeklyVolatility,
        )
        self.assertEqual(len(NoReturnsSkip.DEPENDENCIES), 2)

    def test_with_returns_skip(self) -> None:
        """
        Verify `_RETURNS_LOOKBACK`, `_RETURNS_SKIP` and `_VOLATILITY_LOOKBACK`
        dependencies created.
        """

        class WithReturnsSkip(RiskAdjustedReturns):
            ID = "with_returns_skip"
            CATEGORY = "testing_dependencies"
            RETURNS_LOOKBACK = Window.WEEK
            RETURNS_SKIP = Window.DAY
            VOLATILITY_LOOKBACK = Window.WEEK
            HORIZONS = set()

        self.assertIs(WithReturnsSkip._RETURNS_LOOKBACK, features.WeeklyReturns)
        self.assertIs(WithReturnsSkip._RETURNS_SKIP, features.DailyReturns)
        self.assertIs(
            WithReturnsSkip._VOLATILITY_LOOKBACK,
            features.WeeklyVolatility,
        )
        self.assertEqual(len(WithReturnsSkip.DEPENDENCIES), 3)


class TestRiskAdjustedCompute(RegistryIsolatedTestCase):
    REGISTRY_OWNER = Alpha

    T = 5000
    N = 500

    def test_compute(self) -> None:
        """
        Verify `compute` returns the returns over `RETURNS_LOOKBACK` or between
        `RETURNS_LOOKBACK` and `RETURNS_SKIP`, divided by the rolling volatility
        over `VOLATILITY_LOOKBACK`.
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
        windows_vol_pairs: list[tuple[Window, type[features.Volatility]]] = [
            (Window.DAY, features.DailyVolatility),
            (Window.WEEK, features.WeeklyVolatility),
            (Window.MONTH, features.MonthlyVolatility),
            (Window.QUARTER, features.QuarterlyVolatility),
            (Window.HALF_YEAR, features.HalfYearlyVolatility),
            (Window.YEAR, features.YearlyVolatility),
        ]

        for returns_lookback, lookback_returns in windows_return_pairs:
            for volatility_lookback, volatility in windows_vol_pairs:

                class RiskAdjustedNoReturnsSkip(RiskAdjustedReturns):
                    ID = (
                        f"returns_lookback_{returns_lookback}_volatility_"
                        f"lookback_{volatility_lookback}"
                    )
                    CATEGORY = "testing_compute"
                    RETURNS_LOOKBACK = returns_lookback
                    VOLATILITY_LOOKBACK = volatility_lookback
                    HORIZONS = set()

                returns_over_lookback = rng.uniform(
                    0,
                    10,
                    size=(TestRiskAdjustedCompute.T, TestRiskAdjustedCompute.N),
                ).astype(md.Scalar)
                nan_mask = rng.uniform(size=returns_over_lookback.shape) < 0.1
                returns_over_lookback[nan_mask] = np.nan

                volatility_over_lookback = rng.uniform(
                    0,
                    10,
                    size=(TestRiskAdjustedCompute.T, TestRiskAdjustedCompute.N),
                ).astype(md.Scalar)
                nan_mask = rng.uniform(size=returns_over_lookback.shape) < 0.1
                volatility_over_lookback[nan_mask] = np.nan

                x = {
                    lookback_returns.ID: returns_over_lookback,
                    volatility.ID: volatility_over_lookback,
                }
                np.testing.assert_array_equal(
                    RiskAdjustedNoReturnsSkip.compute(x),
                    returns_over_lookback / volatility_over_lookback,
                )

                for returns_skip, skip_returns in windows_return_pairs:
                    if not returns_skip < returns_lookback:
                        continue

                    class RiskAdjustedWithReturnsSkip(RiskAdjustedReturns):
                        ID = (
                            f"returns_lookback_{returns_lookback}_returns_skip_"
                            f"{returns_skip}_volatility_lookback_"
                            f"{volatility_lookback}"
                        )
                        CATEGORY = "testing_compute"
                        RETURNS_LOOKBACK = returns_lookback
                        RETURNS_SKIP = returns_skip
                        VOLATILITY_LOOKBACK = volatility_lookback
                        HORIZONS = set()

                    returns_over_skip = rng.uniform(
                        0,
                        10,
                        size=(
                            TestRiskAdjustedCompute.T,
                            TestRiskAdjustedCompute.N,
                        ),
                    ).astype(md.Scalar)
                    nan_mask = rng.uniform(size=returns_over_skip.shape) < 0.1
                    returns_over_skip[nan_mask] = np.nan

                    x = {
                        lookback_returns.ID: returns_over_lookback,
                        skip_returns.ID: returns_over_skip,
                        volatility.ID: volatility_over_lookback,
                    }
                    np.testing.assert_array_equal(
                        RiskAdjustedWithReturnsSkip.compute(x),
                        (
                            (returns_over_lookback - returns_over_skip) /
                            volatility_over_lookback
                        ),
                    )

if __name__ == "__main__":
    unittest.main()
