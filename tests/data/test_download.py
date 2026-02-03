import json
import shutil
import unittest
from pathlib import Path

import pandas as pd

from alpha_research_framework import download
from alpha_research_framework.data import metadata_path, stocks_path
from alpha_research_framework.data.structure import log_path

DEST = Path("download_test")
START_DATE = "2020-01-01"
END_DATE = "2021-01-01"
TICKERS = {
    # Traded throughout date range
    "AAPL": {
        "has_data": True,
        "first_trading_day": pd.to_datetime("2020-01-02"),
        "last_trading_day": pd.to_datetime("2020-12-31"),
        "first_close": 75.0875015258789,
        "last_close": 132.69000244140625
    },
    # IPO'd in date range
    "SNOW": {
        "has_data": True,
        "first_trading_day": pd.to_datetime("2020-09-16"),
        "last_trading_day": pd.to_datetime("2020-12-31"),
        "first_close": 253.92999267578125,
        "last_close": 281.3999938964844
    },
    # Delisted in date range
    "LK": {"has_data": False},
    # Non-existent ticker
    "BMX": {"has_data": False},
}


class TestDownload(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        """Download test data."""
        download(DEST, TICKERS.keys(), START_DATE, END_DATE)

    @classmethod
    def tearDownClass(cls) -> None:
        """Remove test data."""
        shutil.rmtree(DEST, ignore_errors=True)

    def test_download_log(self) -> None:
        """Verify download log against known outcomes."""

        with (log_path(DEST)).open() as f:
            download_log = json.load(f)

        for ticker, info in TICKERS.items():
            if info["has_data"]:
                self.assertEqual(download_log[ticker], "success")
            else:
                self.assertEqual(download_log[ticker], "no_data")

    def test_metadata(self) -> None:
        """Verify metadata."""

        with (metadata_path(DEST)).open() as f:
            metadata = json.load(f)

        self.assertEqual(metadata["start_date"], START_DATE)
        self.assertEqual(metadata["end_date"], END_DATE)

        for ticker, info in TICKERS.items():
            if info["has_data"]:
                self.assertIn(ticker, metadata["tickers"].keys())
            else:
                self.assertNotIn(ticker, metadata["tickers"].keys())
        
    def test_raw(self) -> None:
        """Verify raw stock data against known data."""

        for ticker, info in TICKERS.items():
            if not info["has_data"]:
                continue
            df = pd.read_parquet(stocks_path(DEST) / f"{ticker}.parquet")
            self.assertEqual(df.index[0], info["first_trading_day"])
            self.assertEqual(df.index[-1], info["last_trading_day"])
            self.assertEqual(df["close"].iloc[0], info["first_close"])
            self.assertEqual(df["close"].iloc[-1], info["last_close"])


if __name__ == "__main__":
    unittest.main()
