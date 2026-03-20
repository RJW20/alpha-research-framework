import json
import shutil
import unittest
from pathlib import Path

import pandas as pd

from alpha_research_framework import download
from alpha_research_framework.download import metadata_path, stocks_path
from alpha_research_framework.download.structure import log_path

DESTINATION = Path("download_test")
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
START_DATE = "2020-01-01"
YEARS = 1


class TestDownload(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        """Download test data."""
        download(DESTINATION, TICKERS.keys(), START_DATE, YEARS)

    @classmethod
    def tearDownClass(cls) -> None:
        """Remove test data."""
        shutil.rmtree(DESTINATION, ignore_errors=True)

    def test_end_date(self) -> None:
        """Verify end_date cannot be beyond the current date."""

        with self.assertRaises(ValueError):
            download(
                DESTINATION,
                TICKERS.keys(),
                pd.Timestamp.now().strftime("%Y-%m-%d"),
                1
            )

    def test_download_log(self) -> None:
        """Verify download log against known outcomes."""

        with (log_path(DESTINATION)).open() as f:
            download_log = json.load(f)

        for ticker, info in TICKERS.items():
            if info["has_data"]:
                self.assertEqual(download_log[ticker], "success")
            else:
                self.assertEqual(download_log[ticker], "no_data")

    def test_metadata(self) -> None:
        """Verify metadata."""

        with (metadata_path(DESTINATION)).open() as f:
            metadata = json.load(f)

        self.assertEqual(metadata["start_date"], START_DATE)
        end_date = (
            pd.to_datetime(START_DATE) + pd.DateOffset(years=YEARS, days=-1)
        ).strftime("%Y-%m-%d")
        self.assertEqual(metadata["end_date"], end_date)

        for ticker, info in TICKERS.items():
            if info["has_data"]:
                self.assertIn(ticker, metadata["tickers"])
            else:
                self.assertNotIn(ticker, metadata["tickers"])
        
    def test_raw(self) -> None:
        """Verify raw stock data against known data."""

        for ticker, info in TICKERS.items():
            if not info["has_data"]:
                continue
            df = pd.read_parquet(stocks_path(DESTINATION) / f"{ticker}.parquet")
            self.assertEqual(df.index[0], info["first_trading_day"])
            self.assertEqual(df.index[-1], info["last_trading_day"])
            self.assertEqual(df["close"].iloc[0], info["first_close"])
            self.assertEqual(df["close"].iloc[-1], info["last_close"])


if __name__ == "__main__":
    unittest.main()
