import unittest

import numpy as np

from alpha_research_framework import Window
from alpha_research_framework.features import FeatureSpec, LogPrice, Returns


class TestReturns(unittest.TestCase):

    def test_compute(self) -> None:
        """
        Verify out is populated with values that are the difference between
        market data prices spaced at multiple lookback window sizes apart.
        """

        for lookback in Window:
            
            T = lookback.value * 10
            returns = Returns(lookback)
            log_prices = np.arange(T, dtype=np.float32)
            features = {FeatureSpec(LogPrice, ()): log_prices}
            out = np.empty_like(log_prices, dtype=np.float32)

            returns.compute({}, features, out)
            
            expected = np.array(
                (
                    [np.nan] * lookback.value +
                    [lookback.value] * (T - lookback.value)
                ), dtype=np.float32
                )
            np.testing.assert_array_equal(out, expected)


if __name__ == "__main__":
    unittest.main()
