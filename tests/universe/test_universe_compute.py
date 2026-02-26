import unittest

import numpy as np
import pandas as pd

import alpha_research_framework.market_data as md
from alpha_research_framework import Universe, Window


class TestComputeMarketData(unittest.TestCase):

    def test(self) -> None:
        """Verify `price = adj_close` and `volume = volume / adj_factor`."""

        stock_data = pd.DataFrame({
            "adj_close": [10.0, 11.0, 12.0],
            "volume": [100, 200, 300],
            "adj_factor": [1.0, 2.0, 3.0],
        })
        price, volume = Universe._compute_market_data(stock_data)
        np.testing.assert_array_equal(
            price,
            np.array([10.0, 11.0, 12.0], dtype=md.Scalar),
        )
        np.testing.assert_array_equal(
            volume,
            np.array([100.0, 100.0, 100.0], dtype=md.Scalar),
        )


class TestUniverseComputeMask(unittest.TestCase):

    def test_all_conditions_met(self) -> None:
        """Verify a stock that meets all requirements is never masked out."""

        stock_data = pd.DataFrame({
            "adj_close": [10.0, np.nan, 10.0],
            "volume": [1_000, 1_000, 1_000],
        })
        mask = Universe._compute_mask(
            stock_data=stock_data,
            shares=1_000_000,
            liquidity_threshold=5_000,
            mcap_threshold=5_000_000,
            lookback=Window.WEEK,
        )
        self.assertTrue(mask.all())

    def test_compute_mask_nan_adj_close(self) -> None:
        """
        Verify a stock with no non NaN `adj_close` over `lookback` is masked
        out.
        """

        stock_data = pd.DataFrame({
            "adj_close": [10.0, np.nan, 10.0],
            "volume": [1_000, 1_000, 1_000],
        })
        mask = Universe._compute_mask(
            stock_data=stock_data,
            shares=1_000_000,
            liquidity_threshold=0,
            mcap_threshold=0,
            lookback=Window.DAY,
        )
        np.testing.assert_array_equal(
            mask,
            np.array([True, False, True]),
        )

    def test_liquidity_threshold(self) -> None:
        """
        Verify a stock that falls below the average liquidity requirement over
        `lookback` is masked out.
        """

        stock_data = pd.DataFrame({
            "adj_close": [10.0, 10.0, 10.0],
            "volume": [10, 100, 1_000],
        })
        mask = Universe._compute_mask(
            stock_data=stock_data,
            shares=1_000_000,
            liquidity_threshold=1_000,
            mcap_threshold=0,
            lookback=Window.DAY,
        )
        np.testing.assert_array_equal(
            mask,
            np.array([False, True, True]),
        )

    def test_mcap_threshold(self) -> None:
        """
        Verify a stock that falls below the market cap requirement is masked
        out.
        """

        stock_data = pd.DataFrame({
            "adj_close": [1.0, 10.0, 100.0],
            "volume": [1_000, 1_000, 1_000],
        })
        mask = Universe._compute_mask(
            stock_data=stock_data,
            shares=1_000_000,
            liquidity_threshold=1_000,
            mcap_threshold=10_000_000,
            lookback=Window.DAY,
        )
        np.testing.assert_array_equal(
            mask,
            np.array([False, True, True]),
        )


if __name__ == "__main__":
    unittest.main()
