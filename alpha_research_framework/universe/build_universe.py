import os
import shutil
from functools import partial
from graphlib import TopologicalSorter
from pathlib import Path
from typing import Iterable, TypeVar

import numpy as np

import alpha_research_framework.alphas as alphas
import alpha_research_framework.features as features
import alpha_research_framework.market_data as md
import alpha_research_framework.observables as observables
from alpha_research_framework.download.metadata import TickerInfo
from alpha_research_framework.registrable import resolve
from alpha_research_framework.scalar import Scalar
from alpha_research_framework.window import Window

from .calendar import Calendar
from .equity_data import EquityData, TickerData
from .sector import Industry, Sector
from .universe import Universe

# ------------------------------------------------------------------------------
# Feature/Observable deduction
# ------------------------------------------------------------------------------

def _deduce_cross_sectional_features(
    alphas: Iterable[type[alphas.Alpha]],
) -> set[type[features.Feature]]:
    """
    Return a set containing all the features that are required in a cross-
    section to compute every alpha in `alphas`.
    """

    xs_features: set[type[features.Feature]] = set()
    xs_features = xs_features.union(*[a.REQUIRED_FEATURES for a in alphas])
    return xs_features

def _scan_feature_tree(
    features_: Iterable[type[features.Feature]],
) -> tuple[set[type[observables.Observable]], set[type[features.Feature]]]:
    """
    Return 2 sets containing all features present in the reverse linked
    lists defined by `DerivedFeature.SOURCE` and all observables defined by
    `PrimitiveFeature.OBSERVABLE` at the heads of those lists.
    """

    observable_set: set[type[observables.Observable]] = set()
    feature_set: set[type[features.Feature]] = set()
    for feature in features_:
        current = feature
        while current not in feature_set:
            feature_set.add(current)
            if issubclass(current, features.DerivedFeature):
                current = current.SOURCE
            else:
                observable_set.add(current.OBSERVABLE)                      # type: ignore
                break
    return observable_set, feature_set

# ------------------------------------------------------------------------------
# Disk management
# ------------------------------------------------------------------------------

def _make_fresh_directory(path: Path) -> None:
    """
    Remove any file or directory `path` refers to and then create a new
    directory.
    """

    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True)

T = TypeVar('T')
def _allocate_storage(
    identifiers: Iterable[type[T]],
    *,
    path: Path,
    shape: tuple[int, int],
) -> dict[type[T], md.Array]:
    """
    Return a dictionary containing a mapping from `identifier` to a new
    `md.Array` at `path/{identifier.__name__}.dat` with given `shape` for
    every `identifier` in `identifiers`.
    """

    return {
        i: np.memmap(
            path / f"{i.__name__}.dat",
            dtype=Scalar,
            mode="w+",
            shape=shape,
        )
        for i in identifiers
    }

def _release_storage(
    storage: dict[type[T], md.Array],
    *,
    identifiers: Iterable[type[T]] | None = None,
) -> None:
    """
    Remove the `identifier` and release its corresponding `md.Array` for every
    `identifier` in `identifiers`.

    If `identifiers` is not given then the entire `storage` will be released.
    """

    if identifiers is None:
        identifiers = storage.keys()
    for identifier in identifiers:
        arr = storage.pop(identifier)
        if arr.filename:
            os.remove(arr.filename)

# ------------------------------------------------------------------------------
# Market Data and Mask building
# ------------------------------------------------------------------------------

def _populate_market_data(
    market_data: md.MarketData,
    column: int,
    ticker_data: TickerData,
) -> None:
    for obs, values in market_data.items():
        values[:, column] = ticker_data[obs.NAME].to_numpy(dtype=Scalar)

def _populate_mask(
    mask: np.memmap,
    column: int,
    ticker_info: TickerInfo,
    ticker_data: TickerData,
    *,
    liquidity_threshold: float,
    mcap_threshold: float,
    lookback: Window,
) -> None:
    
    exists = ~ticker_data["adj_close"].isna()
    rolling_exists = (
        exists
        .rolling(lookback.value, min_periods=1)
        .max()
        .astype(bool)
    )

    dollar_vol = ticker_data["adj_close"] * ticker_data["volume"]
    liquidity_mask = (
        dollar_vol
        .rolling(lookback.value, min_periods=1)
        .mean()
        >= liquidity_threshold
    )

    rolling_adj_close = (
        ticker_data["adj_close"]
        .rolling(lookback.value, min_periods=1)
        .mean()
    )
    shares = ticker_info["shares_outstanding"]
    mcap_mask = rolling_adj_close * shares >= mcap_threshold

    mask[:, column] = (
        rolling_exists.to_numpy() &
        liquidity_mask.to_numpy() &
        mcap_mask.to_numpy()
    )

def _populate(
    market_data: md.MarketData,
    mask: np.memmap,
    *,
    equity_data: EquityData,
    liquidity_threshold: float,
    mcap_threshold: float,
    lookback: Window,
) -> None:
    """
    Populate `market_data` with values from `equity_data` and build the
    in-universe `mask`.

    `market_data` will contain values for each of its observables at every
    timestamp for each ticker in `equity_data`.
    `mask` describes if the existence (non NaN `adj_close` sometime over
    `lookback`), average liquidity and market cap requirements are met at each
    timestamp for each ticker in `equity_data`.
    """

    populate_market_data = partial(_populate_market_data, market_data)
    populate_mask = partial(
        _populate_mask,
        mask,
        liquidity_threshold=liquidity_threshold,
        mcap_threshold=mcap_threshold,
        lookback=lookback,
    )

    for column, ticker in enumerate(equity_data.tickers):
        ticker_info, ticker_data = equity_data[ticker]
        populate_market_data(column, ticker_data)
        populate_mask(column, ticker_info, ticker_data)

    for mmap in market_data.values():
        mmap.flush()
    mask.flush()

# ------------------------------------------------------------------------------
# Feature Array calculation
# ------------------------------------------------------------------------------

def _order(
    features_: Iterable[type[features.Feature]]
) -> Iterable[type[features.Feature]]:
    features_and_source = {
        f: (
            [f.SOURCE] if issubclass(f, features.DerivedFeature)
            else []
        )
        for f in features_
    }
    ts = TopologicalSorter(features_and_source)
    return ts.static_order()

def _build(
    features_: dict[type[features.Feature], md.Array],
    *,
    market_data: md.MarketData,
) -> None:
    """
    Build the `md.Array` for every feature in `features_` for every timestamp
    and ticker.
    
    Sorts `features_` topologically via `SOURCE` first such that if any derived
    feature's `SOURCE` is also requested it will be computed before it and thus
    have its values available in the cache.
    """

    ordered_features = _order(features_.keys())
    built_features = features.FeatureCache()
    
    for feature in ordered_features:
        values = features_[feature]
        feature.compute(market_data, built_features, values)
        values.flush()
        built_features[feature] = values

# ------------------------------------------------------------------------------
# Public Universe creation
# ------------------------------------------------------------------------------

def build_universe_for(
    alphas_: list[str | type[alphas.Alpha]],
    *,
    src: Path,
    path: Path,
    sector: Sector | None = None,
    industry: Industry | None = None,
    liquidity_threshold: float = 5e8,
    mcap_threshold: float = 5e6,
    lookback: Window = Window.MONTH,
) -> Universe:
    """
    Build a `Universe` designed for evaluating the listed `alphas_` against.
    
    Parameters
    ----------
    alphas_ : list[str | type[Alpha]]
        List of alphas wanting to be evaluated, given by either `ID` or type
        name. Available options include:
        - `"reversal_1d"` or `Reversal1d`
        - `"momentum_12m_1m"` or `Momentum12m1m`
        - `"volatility_20d"` or `Volatility20d`

        For a full list of available alphas along with information on creating
        custom alphas see the "Alphas" section of the documentation.

    src : Path
        Directory containing equity metadata and per-ticker market data, ideally
        created by `download`.

    path : Path
        Directory to store memory-mapped arrays of market features (wiped if
        already exists).

    sector : Sector, optional
        Sector to exclusively include tickers from.

    industry : Industry, optional
        Industry to exclusively include tickers from (must belong to `sector`,
        cannot be specified if `sector` isn't).

    liquidity_threshold: float
        Required average liquidity (over `lookback`) for a stock to be
        considered in-universe on a given date.

    mcap_threshold : float
        Required average market cap (over `lookback`) for a stock to be
        considered in-universe on a given date.

    lookback : Window, default Window.MONTH
        Size of lookback window used in calculation of in-universe mask.

    Returns
    -------
    Universe
        `Universe` instance containing all market features required to compute
        the listed alphas for stocks meeting the specified `sector` and
        `industry` filtering and `liquidity_` and `mcap_thresholds`, built from
        data found in `src`.

    Raises
    ------
    FileNotFoundError
        If a ticker listed in the metadata within `src` doesn't have a raw stock
        file (only possible if `src` wasn't created purely by `download`).

    ValueError
        If `sector` is invalid, `industry` does not belong to `sector` or
        `industry` is specified but `sector` is not.
    """

    resolved_alphas = {resolve(a, alphas.Alpha) for a in alphas_}
    xs_features = _deduce_cross_sectional_features(resolved_alphas)             # type: ignore
    required_observables, required_features = _scan_feature_tree(xs_features)

    equity_data = EquityData(src, sector, industry)
    calendar = Calendar(equity_data.dates)
    shape = calendar.T, len(equity_data.tickers)

    _make_fresh_directory(path)
    market_data = _allocate_storage(
        required_observables,
        path=path,
        shape=shape,
    )
    mask = np.memmap(
        path / "mask.dat",
        dtype=bool,
        mode="w+",
        shape=shape,
    )
    _populate(
        market_data,
        mask,
        equity_data=equity_data,
        liquidity_threshold=liquidity_threshold,
        mcap_threshold=mcap_threshold,
        lookback=lookback,
    )

    features_ = _allocate_storage(required_features, path=path, shape=shape)
    _build(features_, market_data=market_data)

    _release_storage(market_data)
    _release_storage(features_, identifiers=required_features-xs_features)

    return Universe(shape, calendar, mask, features_)
