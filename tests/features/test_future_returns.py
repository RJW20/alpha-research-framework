import unittest

import numpy as np

import alpha_research_framework.market_data as md
from alpha_research_framework.features import (
    DailyFutureReturns,
    Feature,
    FutureReturns,
    HalfYearlyFutureReturns,
    LogPrice,
    MonthlyFutureReturns,
    QuarterlyFutureReturns,
    WeeklyFutureReturns,
    YearlyFutureReturns,
)
from tests.utils import RegistryIsolatedTestCase


class TestFutureReturnsHorizon(RegistryIsolatedTestCase):
    REGISTRY_OWNER = Feature

    def test_class_var_assertion(self) -> None:
        """
        Verify definition and type of `HORIZON` asserted when subclassing.
        """

        with self.assertRaises(AttributeError):
            class NoHorizon(FutureReturns):
                pass

        with self.assertRaises(TypeError):
            class IncompatibleHorizon(FutureReturns):
                HORIZON = 1


class TestFutureReturnsCompute(RegistryIsolatedTestCase):
    REGISTRY_OWNER = Feature

    T = 5000
    N = 500

    def test_compute(self) -> None:
        """
        Verify `out` is populated with values that are the difference between
        entries in `market_data.prices` at `HORIZON` distance apart.
        """

        rng = np.random.default_rng(0)
        log_prices = rng.uniform(
            0,
            100,
            size=(TestFutureReturnsCompute.T, TestFutureReturnsCompute.N),
        ).astype(md.Scalar)
        nan_mask = rng.uniform(size=log_prices.shape) < 0.1
        log_prices[nan_mask] = np.nan
        features = {LogPrice.ID: log_prices}

        future_returns_features: list[type[FutureReturns]] = [
            DailyFutureReturns,
            WeeklyFutureReturns,
            MonthlyFutureReturns,
            QuarterlyFutureReturns,
            HalfYearlyFutureReturns,
            YearlyFutureReturns,
        ]
        for future_returns in future_returns_features:

            out = np.empty_like(log_prices, dtype=md.Scalar)
            future_returns.compute(None, features, out)
            horizon = future_returns.HORIZON.value
            expected = np.concatenate(
                [
                    log_prices[horizon:] - log_prices[:-horizon],
                    np.full((horizon, TestFutureReturnsCompute.N), np.nan),
                ],
                axis=0
            ).astype(md.Scalar)
            np.testing.assert_array_equal(out, expected)


if __name__ == "__main__":
    unittest.main()
