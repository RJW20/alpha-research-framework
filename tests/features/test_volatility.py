import unittest

import numpy as np
import pandas as pd

from alpha_research_framework import Window
from alpha_research_framework.features import FeatureSpec, Returns, Volatility


class TestVolatility(unittest.TestCase):

    def test_compute(self) -> None:
        """
        Verify out is populated with values that are the rolling standard
        deviation of market data prices.
        """

        N = 10

        for lookback in Window:
            
            T = lookback.value * 10
            volatility = Volatility(lookback)
            rng = np.random.default_rng(0)
            returns = rng.uniform(0, T, (T, N)).astype(np.float32)
            nan_mask = rng.uniform(size=returns.shape) < 0.1
            returns[nan_mask] = np.nan
            features = {FeatureSpec(Returns, Window.DAY): returns}
            out = np.empty_like(returns, dtype=np.float32)

            volatility.compute({}, features, out)
            
            df = pd.DataFrame(returns)
            expected = df.rolling(lookback.value, min_periods=1).std()
            np.testing.assert_array_almost_equal(
                out,
                expected.to_numpy(dtype=np.float32),
                decimal=3
            )


if __name__ == "__main__":
    unittest.main()
