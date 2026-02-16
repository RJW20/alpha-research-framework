import json
from pathlib import Path

import pandas as pd
import pandas_market_calendars as mcal

from alpha_research_framework.equity_data.metadata import Metadata
from alpha_research_framework.equity_data.structure import (
    metadata_path,
    stock_path,
    stocks_path,
)


def _write_metadata(root: Path, metadata: Metadata) -> None:
    with (metadata_path(root)).open("w") as f:
            json.dump(metadata, f)


def _get_dates(start_date: str, end_date: str) -> pd.Index:
    nyse = mcal.get_calendar('NYSE')
    schedule = nyse.schedule(start_date, end_date)
    return schedule.index.astype('datetime64[ms]')


def _write_stock(root: Path, ticker: str, index: pd.Index) -> None:
    t = len(index)
    df = pd.DataFrame(
        {
            "adj_close": [10.0] * t,
            "volume": [1_000] * t,
            "adj_factor": [1.0] * t,
        },
        index=index,
    )
    df.to_parquet(stock_path(root, ticker))

def create_equity_data_dir(dest: Path, metadata: Metadata) -> None:
    """
    Write metadata and dummy stock data for each ticker in metadata to dest.
    """

    _write_metadata(dest, metadata)

    stocks_path(dest).mkdir()
    dates = _get_dates(metadata["start_date"], metadata["end_date"])
    for ticker in metadata["tickers"]:
        _write_stock(dest, ticker, dates)
