import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from alpha_research_framework.download import download
from alpha_research_framework.download.structure import (
    log_path,
    metadata_path,
    stocks_path,
)

TICKERS = {
    # Traded throughout date range
    "AAPL": {
        "has_data": True,
        "first_trading_day": pd.to_datetime("2020-01-02"),
        "last_trading_day": pd.to_datetime("2020-12-31"),
        "first_adj_close": 72.401,
        "last_adj_close": 129.047,
    },
    # IPO'd in date range
    "SNOW": {
        "has_data": True,
        "first_trading_day": pd.to_datetime("2020-09-16"),
        "last_trading_day": pd.to_datetime("2020-12-31"),
        "first_adj_close": 253.930,
        "last_adj_close": 281.400,
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
        """Create temporary directory abd download test data to it."""
        cls.tmp_dir = TemporaryDirectory()
        cls.dest = Path(cls.tmp_dir.name)
        download(cls.dest, TICKERS.keys(), START_DATE, YEARS)

    @classmethod
    def tearDownClass(cls) -> None:
        """Remove test data."""
        cls.tmp_dir.cleanup()

    def test_end_date(self) -> None:
        """Verify `end_date` cannot be beyond the current date."""

        with self.assertRaises(ValueError):
            download(None, None, pd.Timestamp.now().strftime("%Y-%m-%d"), 1)

    def test_download_log(self) -> None:
        """Verify download log against known outcomes."""

        with (log_path(self.dest)).open() as f:
            download_log = json.load(f)

        for ticker, info in TICKERS.items():
            if info["has_data"]:
                self.assertEqual(download_log[ticker], "success")
            else:
                self.assertEqual(download_log[ticker], "no_data")

    def test_metadata(self) -> None:
        """Verify metadata."""

        with (metadata_path(self.dest)).open() as f:
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
            df = pd.read_parquet(stocks_path(self.dest) / f"{ticker}.parquet")
            self.assertEqual(df.index[0], info["first_trading_day"])
            self.assertEqual(df.index[-1], info["last_trading_day"])
            self.assertAlmostEqual(
                df["adj_close"].iloc[0],
                info["first_adj_close"],
                places=3,
            )
            self.assertAlmostEqual(
                df["adj_close"].iloc[-1],
                info["last_adj_close"],
                places=3,
            )


if __name__ == "__main__":
    unittest.main()
