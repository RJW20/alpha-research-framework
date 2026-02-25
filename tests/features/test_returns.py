import unittest

import numpy as np

import alpha_research_framework.market_data as md
from alpha_research_framework.features import (
    DailyReturns,
    Feature,
    HalfYearlyReturns,
    LogPrice,
    MonthlyReturns,
    QuarterlyReturns,
    WeeklyReturns,
    YearlyReturns,
)
from alpha_research_framework.features.returns import Returns
from tests.utils import RegistryIsolatedTestCase


class TestReturnsLookback(RegistryIsolatedTestCase):
    REGISTRY_OWNER = Feature

    def test_class_var_assertion(self) -> None:
        """
        Verify definition and type of `LOOKBACK` asserted when subclassing.
        """

        with self.assertRaises(AttributeError):
            class NoLookback(Returns):
                pass

        with self.assertRaises(TypeError):
            class IncompatibleLookback(Returns):
                LOOKBACK = 1


class TestReturnsCompute(RegistryIsolatedTestCase):
    REGISTRY_OWNER = Feature

    T = 5000
    N = 500

    def test_compute(self) -> None:
        """
        Verify `out` is populated with values that are the difference between
        entries in `market_data.prices` at `LOOKBACK` distance apart.
        """

        rng = np.random.default_rng(0)
        log_prices = rng.uniform(
            0,
            100,
            size=(TestReturnsCompute.T, TestReturnsCompute.N),
        ).astype(md.Scalar)
        nan_mask = rng.uniform(size=log_prices.shape) < 0.1
        log_prices[nan_mask] = np.nan
        features = {LogPrice.ID: log_prices}

        returns_features: list[type[Returns]] = [
            DailyReturns,
            WeeklyReturns,
            MonthlyReturns,
            QuarterlyReturns,
            HalfYearlyReturns,
            YearlyReturns,
        ]
        for returns in returns_features:
         
            out = np.empty_like(log_prices, dtype=md.Scalar)
            returns.compute(None, features, out)
            lookback = returns.LOOKBACK.value
            expected = np.concatenate(
                [
                    np.full((lookback, TestReturnsCompute.N), np.nan),
                    log_prices[lookback:] - log_prices[:-lookback],
                ],
                axis=0
            ).astype(md.Scalar)
            np.testing.assert_array_equal(out, expected)


if __name__ == "__main__":
    unittest.main()
