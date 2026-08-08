from functools import partial
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

import alpha_research_framework.alphas as alphas
import alpha_research_framework.market_data as md
import alpha_research_framework.observables as observables
import alpha_research_framework.series as series
import alpha_research_framework.signals as signals
from alpha_research_framework.download.metadata import TickerInfo
from alpha_research_framework.registrable import resolve
from alpha_research_framework.scalar import Scalar
from alpha_research_framework.window import Window

from .calendar import Calendar
from .equity_data import EquityData, TickerData
from .sector import Industry, Sector
from .universe import Universe

# ------------------------------------------------------------------------------
# Series/Observable deduction
# ------------------------------------------------------------------------------

def _deduce_cross_sectional_series(
    signals_: Iterable[type[signals.Signal]],
) -> set[type[series.Series]]:
    """
    Return a set containing all the `Series` that are required in a cross-
    section to compute every `Signal` in `signals_`.
    """

    visited_signals: set[type[signals.Signal]] = set()
    series_set: set[type[series.Series]] = set()

    def find_root(node: type[signals.Signal]) -> None:
        if node in visited_signals:
            return
        visited_signals.add(node)
        try: # Maybe its a SeriesSignal
            series_set.add(node.SERIES)                                         # type: ignore
        except AttributeError:
            try: # Maybe its a NegatedSignal
                find_root(node.SOURCE)                                          # type: ignore
            except AttributeError:
                # Must be a CombinedSignal
                find_root(node.SOURCE_LEFT)                                     # type: ignore
                find_root(node.SOURCE_RIGHT)                                    # type: ignore

    for s in signals_:
        find_root(s)

    return series_set

def _deduce_forward_returns_series(
    horizons: Iterable[Window]
) -> set[type[series.Series]]:
    """
    Return a set containing all the forward returns `Series` that pertain to
    a `Window` in `horizons`.
    """

    window_to_forward_returns = {
        Window.DAY:         series.ForwardReturns1d,
        Window.WEEK:        series.ForwardReturns5d,
        Window.MONTH:       series.ForwardReturns20d,
        Window.QUARTER:     series.ForwardReturns63d,
        Window.HALF_YEAR:   series.ForwardReturns126d,
        Window.YEAR:        series.ForwardReturns252d,
    }
    return set(window_to_forward_returns[h] for h in horizons)

def _deduce_market_data_observables(
    series_: Iterable[type[series.Series]],
) -> set[type[observables.Observable]]:
    """
    Return a set containing all the `Observable`s that are required in the
    market data to compute every `Series` in `series_`.
    """

    visited_series: set[type[series.Series]] = set()
    observable_set: set[type[observables.Observable]] = set()

    def find_root(node: type[series.Series]) -> None:
        if node in visited_series:
            return
        visited_series.add(node)
        try: # Maybe its an ObservableSeries
            observable_set.add(node.OBSERVABLE)                                 # type: ignore
        except AttributeError:
            try: # Maybe its a TransformedSeries
                find_root(node.SOURCE)                                          # type: ignore
            except AttributeError:
                # Must be a CombinedSeries
                find_root(node.SOURCE_LEFT)                                     # type: ignore
                find_root(node.SOURCE_RIGHT)                                    # type: ignore

    for s in series_:
        find_root(s)

    return observable_set

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

# ------------------------------------------------------------------------------
# Public Universe creation
# ------------------------------------------------------------------------------

def build_universe_for(
    alphas_: Sequence[str | type[alphas.Alpha]],
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
    alphas_ : Sequence[str | type[Alpha]]
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
        Directory to store memory-mapped arrays of market series (wiped if
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
        `Universe` instance containing all market series required to compute
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
    xs_series = \
        _deduce_cross_sectional_series(
            [a.SIGNAL for a in resolved_alphas]
        ) and \
        _deduce_forward_returns_series(
            set().union(*[a.HORIZONS for a in resolved_alphas])                 # type: ignore
        )
    md_observables =  _deduce_market_data_observables(xs_series)

    equity_data = EquityData(src, sector, industry)
    calendar = Calendar(equity_data.dates)
    shape = calendar.T, len(equity_data.tickers)

    allocator = md.Allocator(path, shape)
    market_data: md.MarketData = {
        o: allocator.allocate(identifier=o.__name__)
        for o in md_observables
    }
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

    series_ = series.build(
        xs_series,
        market_data=market_data,
        allocator=allocator,
    )

    for arr in set(market_data.values()) - set(series_.values()):
        allocator.release(arr)

    return Universe(shape, calendar, mask, series_)
