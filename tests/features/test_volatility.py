import unittest

import numpy as np
import pandas as pd

import alpha_research_framework.market_data as md
from alpha_research_framework.features import (
    DailyReturns,
    DailyVolatility,
    Feature,
    HalfYearlyVolatility,
    MonthlyVolatility,
    QuarterlyVolatility,
    WeeklyVolatility,
    YearlyVolatility,
)
from alpha_research_framework.features.volatility import Volatility
from tests.utils import RegistryIsolatedTestCase


class TestVolatilityLookback(RegistryIsolatedTestCase):
    REGISTRY_OWNER = Feature

    def test_class_var_assertion(self) -> None:
        """
        Verify definition and type of `LOOKBACK` asserted when subclassing.
        """

        with self.assertRaises(AttributeError):
            class NoLookback(Volatility):
                pass

        with self.assertRaises(TypeError):
            class IncompatibleLookback(Volatility):
                LOOKBACK = 1


class TestVolatilityCompute(RegistryIsolatedTestCase):
    REGISTRY_OWNER = Feature

    T = 5000
    N = 500

    def test_compute(self) -> None:
        """
        Verify `out` is populated with values that are the rolling standard
        deviation of `market_data.prices`.
        """

        rng = np.random.default_rng(0)
        returns = rng.uniform(
            0,
            10,
            size=(TestVolatilityCompute.T, TestVolatilityCompute.N),
        ).astype(md.Scalar)
        nan_mask = rng.uniform(size=returns.shape) < 0.1
        returns[nan_mask] = np.nan
        features = {DailyReturns.ID: returns}

        volatility_features: list[type[Volatility]] = [
            DailyVolatility,
            WeeklyVolatility,
            MonthlyVolatility,
            QuarterlyVolatility,
            HalfYearlyVolatility,
            YearlyVolatility,
        ]
        for volatility in volatility_features:

            out = np.empty_like(returns, dtype=md.Scalar)
            volatility.compute(None, features, out)
            expected = (
                pd.DataFrame(returns)
                .rolling(volatility.LOOKBACK.value, min_periods=1)
                .std()
                .to_numpy(dtype=md.Scalar)
            )
            np.testing.assert_array_almost_equal(out, expected)


if __name__ == "__main__":
    unittest.main()
