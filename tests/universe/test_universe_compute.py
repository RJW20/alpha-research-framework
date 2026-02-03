import unittest

import numpy as np
import pandas as pd

from alpha_research_framework import Universe, Window


class TestUniverseCompute(unittest.TestCase):

    def test_compute_market_data(self) -> None:

        df = pd.DataFrame({
            "adj_close": [10.0, 11.0, 12.0],
            "volume": [100, 200, 300],
            "adj_factor": [1.0, 2.0, 3.0],
        })

        adj_close, adj_volume = Universe._compute_market_data(df)

        np.testing.assert_array_equal(
            adj_close,
            np.array([10.0, 11.0, 12.0], dtype=np.float32),
        )

        np.testing.assert_array_equal(
            adj_volume,
            np.array([100.0, 100.0, 100.0], dtype=np.float32),
        )

    def test_compute_mask_all_conditions_met(self) -> None:

        df = pd.DataFrame({
            "adj_close": [10.0, 10.0, 10.0],
            "volume": [1_000, 1_000, 1_000],
        })

        mask = Universe._compute_mask(
            df=df,
            shares=1_000_000,
            liquidity_threshold=5_000,
            mcap_threshold=5_000_000,
            lookback=Window.DAY,
        )

        self.assertTrue(mask.all())

    def test_compute_mask_nan_adj_close(self) -> None:

        df = pd.DataFrame({
            "adj_close": [10.0, np.nan, 10.0],
            "volume": [1_000, 1_000, 1_000],
        })

        mask = Universe._compute_mask(
            df=df,
            shares=1_000_000,
            liquidity_threshold=0,
            mcap_threshold=0,
            lookback=Window.DAY,
        )

        np.testing.assert_array_equal(
            mask,
            np.array([True, False, True]),
        )

    def test_compute_mask_liquidity_threshold(self) -> None:

        df = pd.DataFrame({
            "adj_close": [10.0, 10.0, 10.0],
            "volume": [10, 100, 1_000],
        })

        mask = Universe._compute_mask(
            df=df,
            shares=1_000_000,
            liquidity_threshold=1_000,
            mcap_threshold=0,
            lookback=Window.DAY,
        )

        np.testing.assert_array_equal(
            mask,
            np.array([False, True, True]),
        )

    def test_compute_mask_mcap_threshold(self) -> None:

        df = pd.DataFrame({
            "adj_close": [1.0, 10.0, 100.0],
            "volume": [1_000, 1_000, 1_000],
        })

        mask = Universe._compute_mask(
            df=df,
            shares=1_000_000,
            liquidity_threshold=1_000,
            mcap_threshold=10_000_000,
            lookback=Window.DAY,
        )

        np.testing.assert_array_equal(
            mask,
            np.array([False, True, True]),
        )

    def test_compute_mask_rolling_lookback(self) -> None:

        df = pd.DataFrame({
            "adj_close": [10.0, np.nan, 10.0],
            "volume": [1_000, 1_000, 1_000],
        })

        mask = Universe._compute_mask(
            df=df,
            shares=1_000_000,
            liquidity_threshold=0,
            mcap_threshold=0,
            lookback=Window.WEEK,
        )

        self.assertTrue(mask.all())


if __name__ == "__main__":
    unittest.main()
