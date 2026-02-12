import unittest

import numpy as np

from alpha_research_framework import Window
from alpha_research_framework.features import (
    FeatureSpec,
    FutureReturns,
    LogPrice,
)


class TestFutureReturns(unittest.TestCase):

    def test_compute(self) -> None:
        """
        Verify out is populated with values that are the difference between
        market data prices spaced at multiple horizon window sizes apart.
        """

        for horizon in Window:
            
            T = horizon.value * 10
            future_returns = FutureReturns(horizon)
            log_prices = np.arange(T, dtype=np.float32)
            features = {FeatureSpec(LogPrice, ()): log_prices}
            out = np.empty_like(log_prices, dtype=np.float32)

            future_returns.compute({}, features, out)
            
            expected = np.array(
                (
                    [horizon.value]
                    * (T - horizon.value - FutureReturns._ENTRY_LAG)
                    + [np.nan] * (horizon.value + FutureReturns._ENTRY_LAG)
                ),
                dtype=np.float32,
            )
            np.testing.assert_array_equal(out, expected)


if __name__ == "__main__":
    unittest.main()
