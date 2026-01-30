import json
import shutil
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_research_framework.data import metadata_path, stocks_path
from alpha_research_framework.universe.calendar import Calendar
from alpha_research_framework.universe.data_wrapper import DataWrapper
from alpha_research_framework.universe.market_data import MarketData


class Universe:
    """
    Cross-sectional universe backed by memory-mapped arrays.

    Holds a fixed set of tickers, a global calendar, time * stock market data
    (e.g. prices, volumes) and a time-varying in-universe mask.
    The universe is constructed once from on-disk data and is read-only
    thereafter.
    """

    PATH = Path("universe")

    def __init__(
        self,
        src: Path,
        liquidity_threshold: float,
        mcap_threshold: float,
        lookback: int = 20
    ) -> None:
        """
        Initialize the universe and create all required memmapped arrays.

        Loads per-stock data from src, constructs a global trading calendar,
        writes time * stock arrays to disk, and computes the in-universe mask
        using existence, liquidity, and market cap filters.
        """

        shutil.rmtree(self.PATH, ignore_errors=True)
        self.PATH.mkdir(parents=True, exist_ok=False)

        with (metadata_path(src)).open() as f:
            self._metadata = json.load(f)
        
        self._calendar = Calendar(
            self._metadata["start_date"], self._metadata["end_date"]
        )

        tickers = list(self._metadata["tickers"].keys())
        self.N = len(tickers)

        # Allocate memmaps
        shape = (self._calendar.T, self.N)
        self._market_data = MarketData(self.PATH, shape=shape)
        self._mask = np.memmap(
            self.PATH / "mask.dat",
            dtype=bool,
            mode="w+",
            shape=shape,
        )

        # Fill memmaps
        for i, ticker in enumerate(tickers):
            df = pd.read_parquet(stocks_path(src) / f"{ticker}.parquet")
            df = df.reindex(self._calendar.index)

            # Fill memmaps
            self._market_data.adj_close[:,i] = \
                df["adj_close"].astype(np.float32)
            self._market_data.adj_volume[:,i] = \
                df["volume"].astype(np.float32) / \
                df["adj_factor"].astype(np.float32)

            exists = ~df["adj_close"].isna()
            rolling_exists = (
                exists
                .rolling(lookback, min_periods=1)
                .max()
                .astype(np.bool)
            )

            dollar_vol = df["adj_close"] * df["volume"]
            liquidity_mask = (
                dollar_vol
                .rolling(lookback, min_periods=1)
                .mean()
                >= liquidity_threshold
            ).astype(np.bool)

            shares = self._metadata["tickers"][ticker]["shares_outstanding"]
            mcap_mask = (df["adj_close"] * shares >= mcap_threshold).values

            self._mask[:,i] = rolling_exists & liquidity_mask & mcap_mask

        # Flush memmaps
        self._market_data.flush()
        self._mask.flush()

        # Prepare cross-section constituents
        self.features = DataWrapper()
        self.future_returns = DataWrapper()

    def __del__(self) -> None:
        """Remove all memmapped arrays."""

        shutil.rmtree(self.PATH, ignore_errors=True)

    def build_features(self, features: Iterable[str]) -> None:
        pass

    def build_future_returns(self, horizons: Iterable[int]) -> None:
        pass

    def cross_section(self, t: int) -> pd.DataFrame:
        pass
