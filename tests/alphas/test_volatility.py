import unittest

import numpy as np

import alpha_research_framework.features as features
import alpha_research_framework.market_data as md
from alpha_research_framework import Window
from alpha_research_framework.alphas import Alpha
from alpha_research_framework.alphas.volatility import Volatility
from tests.utils import RegistryIsolatedTestCase


class TestVolatilityLookback(RegistryIsolatedTestCase):
    REGISTRY_OWNER = Alpha

    def test_class_var_assertion(self) -> None:
        """
        Verify definition and type of `LOOKBACK` asserted when subclassing.
        """

        with self.assertRaises(AttributeError):
            class NoLookback(Volatility):
                ID = "no_lookback"
                CATEGORY = "testing_lookback"

        with self.assertRaises(TypeError):
            class IncompatibleLookback(Volatility):
                ID = "incompatible_lookback"
                CATEGORY = "testing_lookback"
                LOOKBACK = 1


class TestVolatilityDependencies(RegistryIsolatedTestCase):
    REGISTRY_OWNER = Alpha

    def test_volatility_lookback(self) -> None:
        """Verify `_VOLATILITY_LOOKBACK` dependency created."""

        class NoSkip(Volatility):
            ID = "no_skip"
            CATEGORY = "testing_dependencies"
            LOOKBACK = Window.DAY
            HORIZONS = set()

        self.assertIs(NoSkip._VOLATILITY_LOOKBACK, features.DailyVolatility)
        self.assertEqual(len(NoSkip.DEPENDENCIES), 1)


class TestVolatilityCompute(RegistryIsolatedTestCase):
    REGISTRY_OWNER = Alpha

    T = 5000
    N = 500

    def test_compute(self) -> None:
        """
        Verify compute returns the negative of the rolling volatility over
        `LOOKBACK`.
        """

        rng = np.random.default_rng(0)
        windows_vol_pairs: list[tuple[Window, type[features.Volatility]]] = [
            (Window.DAY, features.DailyVolatility),
            (Window.WEEK, features.WeeklyVolatility),
            (Window.MONTH, features.MonthlyVolatility),
            (Window.QUARTER, features.QuarterlyVolatility),
            (Window.HALF_YEAR, features.HalfYearlyVolatility),
            (Window.YEAR, features.YearlyVolatility),
        ]
        for lookback, volatility in windows_vol_pairs:

            class VolatilityOverLookback(Volatility):
                ID = f"volatility_over_{lookback}"
                CATEGORY = "testing_compute"
                LOOKBACK = lookback
                HORIZONS = set()

            volatility_over_lookback = rng.uniform(
                0,
                10,
                size=(TestVolatilityCompute.T, TestVolatilityCompute.N),
            ).astype(md.Scalar)
            x = {volatility.ID: volatility_over_lookback}
            np.testing.assert_array_equal(
                VolatilityOverLookback.compute(x),
                volatility_over_lookback * -1,
            )


if __name__ == "__main__":
    unittest.main()
