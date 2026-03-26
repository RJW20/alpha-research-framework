import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from alpha_research_framework.download import Metadata, TickerInfo, stock_path
from alpha_research_framework.universe.equity_data.equity_data import EquityData
from alpha_research_framework.universe.sector import (
    INDUSTRIES_PER_SECTOR,
    Industry,
    Sector,
)
from tests.utils import create_download_dir


class TestEquityDataSectorIndustry(unittest.TestCase):

    def test_validate(self) -> None:
        """
        Verify `sector`, `industry` validity for pairs:
        - `None`, `None`
        - `sector`, `None`
        - `sector`, `industry` in `INDUSTRIES_PER_SECTOR[sector]`
        """

        EquityData._validate(None, None)

        for industries in INDUSTRIES_PER_SECTOR.values():
            for industry in industries:
                with self.assertRaises(ValueError):
                    EquityData._validate(None, industry)

        for sector in INDUSTRIES_PER_SECTOR:
            EquityData._validate(sector, None)
            for industries in INDUSTRIES_PER_SECTOR.values():
                if INDUSTRIES_PER_SECTOR[sector] == industries:
                    for industry in industries:
                        EquityData._validate(sector, industry)
                else:
                    for industry in industries:
                        with self.assertRaises(ValueError):
                            EquityData._validate(sector, industry)

    def test_extract_tickers(self) -> None:
        """
        Verify tickers are extracted by `sector` and `industry` when specified.
        """

        tickers: dict[str, TickerInfo] = {
            "AAA": {
                "exchange": "NYSE",
                "currency": "USD",
                "sector": "energy",
                "industry": "thermal coal",
                "shares_outstanding": 100
            },
            "BBB": {
                "exchange": "NYSE",
                "currency": "USD",
                "sector": "energy",
                "industry": "oil & gas drilling",
                "shares_outstanding": 100
            },
            "CCC": {
                "exchange": "NYSE",
                "currency": "USD",
                "sector": "technology",
                "industry": "solar",
                "shares_outstanding": 100
            },
        }

        self.assertEqual(
            EquityData._extract_tickers(tickers, None, None),
            {"AAA", "BBB", "CCC"}
        )
        self.assertEqual(
            EquityData._extract_tickers(tickers, "energy", None),
            {"AAA", "BBB"}
        )
        self.assertEqual(
            EquityData._extract_tickers(tickers, "energy", "thermal coal"),
            {"AAA"}
        )


class TestEquityDataTickersDates(unittest.TestCase):

    METADATA: Metadata = {
        "created_at": "N/A",
        "source": "N/A",
        "start_date": "2020-01-01",
        "end_date": "2020-02-01",
        "tickers": {
            "AAA": {
                "exchange": "NYSE",
                "currency": "USD",
                "sector": "energy",
                "industry": "thermal coal",
                "shares_outstanding": 100
            },
            "BBB": {
                "exchange": "NYSE",
                "currency": "USD",
                "sector": "energy",
                "industry": "oil & gas drilling",
                "shares_outstanding": 100
            },
            "CCC": {
                "exchange": "NYSE",
                "currency": "USD",
                "sector": "technology",
                "industry": "solar",
                "shares_outstanding": 100
            },
        }
    }

    def setUp(self) -> None:
        """
        Create a temporary directory with files to make an `EquityData` instance
        from.
        """

        self._tmp_dir = TemporaryDirectory()
        self.src = Path(self._tmp_dir.name)
        create_download_dir(self.src, self.METADATA)

    def tearDown(self) -> None:
        self._tmp_dir.cleanup()

    def _build_equity_data(
        self,
        sector: Sector | None = None,
        industry: Industry | None = None
    ) -> EquityData:
        return EquityData(self.src, sector, industry)

    def test_assert_all_tickers_exist(self) -> None:
        """Verify a ticker with no stock file causes a `FileNotFoundError`."""

        equity_data = self._build_equity_data()
        stock_path(self.src, "AAA").unlink()
        with self.assertRaises(FileNotFoundError):
            equity_data._assert_all_tickers_exist()

    def test_assert_contains(self) -> None:
        """Verify a ticker not in the equity data causes a `ValueError`."""

        equity_data = self._build_equity_data()
        equity_data._assert_contains("AAA")
        equity_data._assert_contains("BBB")
        equity_data._assert_contains("CCC")
        with self.assertRaises(ValueError):
            equity_data._assert_contains("DDD")

    def test_load_ticker(self) -> None:
        """
        Verify `TickerInfo` and `TickerData` returned are correct and aligned.
        """

        equity_data = self._build_equity_data()
        info, data = equity_data._load_ticker("AAA")

        self.assertEqual(
            info, self.METADATA["tickers"]["AAA"]
        )

        pd.testing.assert_index_equal(data.index, equity_data.dates)
        self.assertEqual(
            data.columns.to_list(),
            ["adj_close", "volume", "adj_factor"]
        )


if __name__ == "__main__":
    unittest.main()
