import unittest

import numpy as np

from alpha_research_framework.features import LogPrice
from tests.features.dummy_market_data import DummyMarketData


class TestLogPrice(unittest.TestCase):

    def test_compute(self) -> None:
        """
        Verify out is populated with values that are the log of the market data
        prices.
        """

        log_price = LogPrice()
        market_data = DummyMarketData(
            np.array([10.0, 11.0, 12.0], dtype=np.float32),
            np.array([], dtype=np.float32)
        )
        out = np.empty_like(market_data.price, dtype=np.float32)

        log_price.compute(market_data, {}, out)

        expected = np.log(market_data.price)
        np.testing.assert_array_equal(out, expected)


if __name__ == "__main__":
    unittest.main()
