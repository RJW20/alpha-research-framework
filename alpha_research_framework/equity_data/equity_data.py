import json
from functools import cached_property
from pathlib import Path

import pandas as pd
import pandas_market_calendars as mcal

from alpha_research_framework.download import (
    Metadata,
    StockInfo,
    metadata_path,
    stock_path,
)
from alpha_research_framework.equity_data.sector import (
    INDUSTRIES_PER_SECTOR,
    Industry,
    Sector,
)


class EquityData:
    """
    Container providing validated, read-only access to equity metadata and
    raw per-ticker market data stored on disk.

    Instances represent a filtered view of the full dataset. If sector and/or
    industry are provided, only matching tickers are exposed.
    """

    def __init__(
        self,
        src: Path,
        sector: Sector | None = None,
        industry: Industry | None = None
    ) -> None:
        """Load the metadata and validate the sector and industry."""

        self._src = src
        self._metadata = self._load_metadata(src)
        self._validate(sector, industry)
        self._sector = sector
        self._industry = industry
        self._tickers = self._extract_tickers(
            self._metadata["tickers"],
            sector,
            industry
        )
        self._assert_all_tickers_exist()

    @cached_property
    def dates(self) -> pd.Index:
        """Return an index containing all dates with valid market data."""

        nyse = mcal.get_calendar('NYSE')
        start_date = self._metadata["start_date"]
        end_date = self._metadata["end_date"]
        schedule = nyse.schedule(start_date, end_date)
        return schedule.index.astype('datetime64[ms]')
    
    @property
    def tickers(self) -> set[str]:
        """Return a set containing all the tickers stored in the EquityData."""
        return self._tickers

    def load_stock(self, ticker: str) -> tuple[StockInfo, pd.DataFrame]:
        """
        Return a tuple containing stock information and a dataframe indexed by
        date holding raw stock data.

        Raises a ValueError if the ticker is not in the EquityData.
        """

        self._assert_contains(ticker)
        info = self._metadata["tickers"][ticker]
        data = pd.read_parquet(
            stock_path(self._src, ticker)
        ).reindex(self.dates)
        return info, data

    @staticmethod
    def _load_metadata(src: Path) -> Metadata:
        """
        Return a dictionary containing metadata about the stocks contained in
        src.
        """
        with (metadata_path(src)).open() as f:
            return json.load(f)
        
    @staticmethod
    def _validate(sector: Sector | None, industry: Industry | None) -> None:
        """
        Raise a ValueError if sector is invalid or industry does not lie within
        sector.
        """

        if sector is not None:
            if sector not in INDUSTRIES_PER_SECTOR:
                raise ValueError(f"Invalid sector: '{sector}'.")
            if (
                industry is not None and
                industry not in INDUSTRIES_PER_SECTOR[sector]
            ):
                raise ValueError(
                    f"Industry '{industry}' does not belong to sector "
                    f"'{sector}'."
                )
        else:
            if industry is not None:
                raise ValueError(
                    "Cannot specify industry if sector is not specified."
                )
            
    @staticmethod
    def _extract_tickers(
        tickers: dict[str, StockInfo],
        sector: Sector | None,
        industry: Industry | None
    ) -> set[str]:
        """
        Return a set containing all tickers that match the sector and industry.

        If the sector is None all tickers will be included.
        Otherwise if the industry is None all tickers from the sector will be
        included.
        """

        if sector is None:
            return set(tickers.keys())
        if industry is None:
            return {
                ticker for ticker, info in tickers.items()
                if info["sector"] == sector
            }
        return {
            ticker for ticker, info in tickers.items()
            if info["sector"] == sector and info["industry"] == industry
        }
            
    def _assert_all_tickers_exist(self) -> None:
        """
        Raise a FileNotFoundError if a ticker doesn't have a raw stock file.
        """

        for ticker in self._tickers:
            path = stock_path(self._src, ticker)
            if not path.exists():
                raise FileNotFoundError(
                    f"Ticker {ticker} found in '{metadata_path(self._src)}' "
                    f"should have a raw stock file '{path}'."
                )

    def _assert_contains(self, ticker: str) -> None:
        """Raise a ValueError if ticker is not in self.tickers."""

        if ticker not in self._tickers:
            if self._sector is not None:
                if self._industry is not None:
                    raise ValueError(
                        f"Ticker {ticker} is not in the EquityData for sector "
                        f"{self._sector} and industry {self._industry}."
                    )
                else:
                    raise ValueError(
                        f"Ticker {ticker} is not in the EquityData for sector "
                        f"{self._sector},"
                    )
            else:
                raise ValueError(
                    f"Ticker {ticker} is not in the EquityData."
                )
