import shutil
from collections.abc import Iterable
from graphlib import TopologicalSorter
from pathlib import Path

import numpy as np
import pandas as pd

import alpha_research_framework.market_data as md
from alpha_research_framework.equity_data import EquityData
from alpha_research_framework.features import Feature, Features, FutureReturns
from alpha_research_framework.universe.calendar import Calendar
from alpha_research_framework.universe.cross_section import CrossSection
from alpha_research_framework.universe.market_data_store import MarketDataStore
from alpha_research_framework.window import Window


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
        path: Path,
        equity_data: EquityData,
        liquidity_threshold: float,
        mcap_threshold: float,
        lookback: Window = Window.MONTH,
    ) -> None:
        """
        Initialise the universe and create all required memmapped arrays.

        Loads per-stock data from `equity_data`, constructs a global trading
        calendar, writes time * stock arrays to disk, and computes the in-
        universe mask using existence, liquidity, and market cap filters.
        """

        self.path = path
        self._prepare_path(path)

        self._calendar = Calendar(equity_data.dates)
        self.shape = (self._calendar.T, len(equity_data.tickers))
        self._market_data, self._mask = self._allocate_storage(path, self.shape)
        for col, ticker in enumerate(equity_data.tickers):
            info, data = equity_data.load_stock(ticker)
            self._market_data[:, col] = self._compute_market_data(data)
            shares = info["shares_outstanding"]
            self._mask[:, col] = self._compute_mask(
                data,
                shares,
                liquidity_threshold,
                mcap_threshold,
                lookback
            )

        self._flush()
        self._features = Features()

    @property
    def dates(self) -> pd.Index:
        """Return all dates with valid market data."""
        return self._calendar.index

    def build_features(self, features: Iterable[type[Feature]]) -> None:
        """
        Build the requested features for every timestamp and stock.
        
        Also builds any features they are dependent on.
        Any features (requested or dependencies) that are already built are
        skipped.
        """

        features = self._expand_dependencies(features)
        features -= set(self._features.keys())
        ordered_features = self._order_dependencies(features)
        
        for feature in ordered_features:
            values = np.memmap(
                self.path / f"{feature.ID}.dat",
                dtype=md.Scalar,
                mode="w+",
                shape=self.shape,
            )
            feature.compute(self._market_data, self._features, values)
            values.flush()
            self._features[feature] = values
        
    def cross_section(self, date: pd.Timestamp) -> CrossSection:
        """
        Return a `CrossSection` containing market data and predictive features
        for all stocks in-universe at the given `date`.
        """

        t = self._calendar.t(date)
        mask = self._mask[t, :]
        return CrossSection(
            {
                "price": self._market_data.price[t, mask],
                "volume": self._market_data.volume[t, mask],
            }
            |
            {
                feature.ID: values[t, mask]
                for feature, values in self._features.items()
                if feature.TAG == Feature.Tag.PREDICTOR
            }
        )
    
    def future_returns(self, date: pd.Timestamp) -> dict[Window, md.Array]:
        """
        Return a `dictionary` mapping horizon to future returns over that
        horizon for all stocks in-universe at the given `date`.
        """

        t = self._calendar.t(date)
        mask = self._mask[t, :]
        return {
            feature.HORIZON: values[t, mask]
            for feature, values in self._features.items()
            if issubclass(feature, FutureReturns)
        }

    # Private helpers

    @staticmethod
    def _prepare_path(path: Path) -> None:
        shutil.rmtree(path, ignore_errors=True)
        path.mkdir(parents=True, exist_ok=False)
    
    @staticmethod
    def _allocate_storage(
        path: Path,
        shape: tuple[int,int]
    ) -> tuple[MarketDataStore, np.memmap]:
        market_data = MarketDataStore(path, shape=shape)
        mask = np.memmap(
            path / "mask.dat",
            dtype=bool,
            mode="w+",
            shape=shape,
        )
        return market_data, mask
    
    @staticmethod
    def _compute_market_data(stock_data: pd.DataFrame) -> tuple[np.ndarray,...]:
        """Return `NumPy` arrays for `price` and `volume`."""

        price = stock_data["adj_close"].to_numpy(dtype=md.Scalar)
        volume = (
            stock_data["volume"] / stock_data["adj_factor"]
        ).to_numpy(dtype=md.Scalar)
        return price, volume

    @staticmethod
    def _compute_mask(
        stock_data: pd.DataFrame,
        shares: int,
        liquidity_threshold: float,
        mcap_threshold: float,
        lookback: Window
    ) -> np.ndarray:
        """
        Return a mask describing when the stock with given `stock_data` meets
        the existence (non NaN `adj_close` sometime over `lookback`), liquidity
        and market cap requirements.
        """

        exists = ~stock_data["adj_close"].isna()
        rolling_exists = (
            exists
            .rolling(lookback.value, min_periods=1)
            .max()
            .astype(bool)
        )

        dollar_vol = stock_data["adj_close"] * stock_data["volume"]
        liquidity_mask = (
            dollar_vol
            .rolling(lookback.value, min_periods=1)
            .mean()
            >= liquidity_threshold
        )

        rolling_adj_close = (
            stock_data["adj_close"]
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

    @staticmethod
    def _expand_dependencies(
        features: Iterable[type[Feature]]
    ) -> set[type[Feature]]:
        """
        Return a set containing all given `features` along with their
        dependencies and their dependencies' dependencies etc.
        """

        expanded: set[type[Feature]] = set()
        to_expand = set(features)
        while to_expand:
            feature = to_expand.pop()
            if feature not in expanded:
                expanded.add(feature)
                to_expand |= feature.DEPENDENCIES
        return expanded
    
    @staticmethod
    def _order_dependencies(
        features: Iterable[type[Feature]]
    ) -> Iterable[type[Feature]]:
        """
        Return an iteratable containing all given `features` in an order such
        that any feature's dependencies come before it.
        """

        features_and_dependencies = {f: f.DEPENDENCIES for f in features}
        ts = TopologicalSorter(features_and_dependencies)
        return ts.static_order()
