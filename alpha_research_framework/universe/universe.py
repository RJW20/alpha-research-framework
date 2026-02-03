import json
import shutil
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_research_framework.data import metadata_path, stocks_path
from alpha_research_framework.universe.calendar import Calendar
from alpha_research_framework.universe.market_data import MarketData
from alpha_research_framework.window import Window

Returns = dict[Window, np.memmap]


class Universe:
    """
    Cross-sectional universe backed by memory-mapped arrays.

    Holds a fixed set of tickers, a global calendar, time * stock market data
    (e.g. prices, volumes) and a time-varying in-universe mask.
    The universe is constructed once from on-disk data and is read-only
    thereafter.
    """

    def __init__(
        self,
        src: Path,
        path: Path,
        liquidity_threshold: float,
        mcap_threshold: float,
        lookback: Window = Window.MONTH,
    ) -> None:
        """
        Initialize the universe and create all required memmapped arrays.

        Loads per-stock data from src, constructs a global trading calendar,
        writes time * stock arrays to disk, and computes the in-universe mask
        using existence, liquidity, and market cap filters.
        """

        self.path = path
        self._prepare_path(path)

        self._metadata = self._load_metadata(src)
        self._calendar = self._build_calendar(self._metadata)

        tickers = list(self._metadata["tickers"].keys())
        self.shape = (self._calendar.T, len(tickers))

        self._market_data, self._mask = self._allocate_storage(path, self.shape)
        
        for col, ticker in enumerate(tickers):
            df = self._load_stock_frame(src, ticker)
            self._market_data[:, col] = self._compute_market_data(df)
            shares = self._metadata["tickers"][ticker]["shares_outstanding"]
            self._mask[:, col] = self._compute_mask(
                df,
                shares,
                liquidity_threshold,
                mcap_threshold,
                lookback
            )

        self._flush()
        self._initialise_cross_section()

    def cross_section(self, t: int) -> pd.DataFrame:
        pass

    def build_future_returns(self, horizons: Iterable[Window]) -> None:
        pass

    # Private helpers

    @staticmethod
    def _prepare_path(path: Path) -> None:
        shutil.rmtree(path, ignore_errors=True)
        path.mkdir(parents=True, exist_ok=False)

    @staticmethod
    def _load_metadata(src: Path) -> dict:
        with (metadata_path(src)).open() as f:
            return json.load(f)

    @staticmethod
    def _build_calendar(metadata: dict) -> Calendar:
        return Calendar(
            metadata["start_date"],
            metadata["end_date"]
        )
    
    @staticmethod
    def _allocate_storage(
        path: Path,
        shape: tuple[int,int]
    ) -> tuple[MarketData, np.memmap]:
        market_data = MarketData(path, shape=shape)
        mask = np.memmap(
            path / "mask.dat",
            dtype=bool,
            mode="w+",
            shape=shape,
        )
        return market_data, mask
    
    def _load_stock_frame(self, src: Path, ticker: str) -> pd.DataFrame:
        df = pd.read_parquet(stocks_path(src) / f"{ticker}.parquet")
        return df.reindex(self._calendar.index)
    
    @staticmethod
    def _compute_market_data(df: pd.DataFrame) -> tuple[np.ndarray,...]:
        adj_close = df["adj_close"].to_numpy(dtype=np.float32)
        adj_volume = (
            df["volume"] / df["adj_factor"]
        ).to_numpy(dtype=np.float32)
        return adj_close, adj_volume

    @staticmethod
    def _compute_mask(
        df: pd.DataFrame,
        shares: int,
        liquidity_threshold: float,
        mcap_threshold: float,
        lookback: Window
    ) -> np.ndarray:

        exists = ~df["adj_close"].isna()
        rolling_exists = (
            exists
            .rolling(lookback.value, min_periods=1)
            .max()
            .astype(bool)
        )

        dollar_vol = df["adj_close"] * df["volume"]
        liquidity_mask = (
            dollar_vol
            .rolling(lookback.value, min_periods=1)
            .mean()
            >= liquidity_threshold
        )

        rolling_adj_close = (
            df["adj_close"]
            .rolling(lookback.value, min_periods=1)
            .mean()
        )
        mcap_mask = rolling_adj_close * shares >= mcap_threshold

        return (
            rolling_exists.to_numpy() &
            liquidity_mask.to_numpy() &
            mcap_mask.to_numpy()
        )

    def _flush(self) -> None:
        self._market_data.flush()
        self._mask.flush()

    def _initialise_cross_section(self) -> None:
        self._features = Features()
        self._future_returns = Returns()
