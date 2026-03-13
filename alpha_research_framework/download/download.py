import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yfinance as yf

from alpha_research_framework.download.metadata import Metadata
from alpha_research_framework.download.structure import (
    log_path,
    metadata_path,
    stock_path,
    stocks_path,
)


def _atomic_write_json(path: Path, data: dict[Any, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    with tmp.open("w") as f:
        json.dump(data, f, indent=2, default=str)
    tmp.replace(path)


def download(
    dest: Path,
    tickers: Iterable[str],
    start_date: str,
    years: int
) -> None:
    """
    Download per-stock daily data and create static metadata.

    Parameters
    ----------
    dest : Path
        Directory to write all downloaded/created files to.
    tickers : Iterable[str]
        Listing of all tickers to download stock data for.
    start_date : str (yyyy-mm-dd)
        First date to retrieve stock data.
    years : int
        Number of years to download daily stock data over.

    Returns
    -------
    None

    Raises
    ------
    pd.ParserError
        When parsing a date from `start_date` fails.
    ValueError
        If a duration of `years` from `start_date` exceeds the current date.

    Side Effects
    ------------
    Writes the following files to `dest`:

    stocks/{ticker}.parquet : pd.DataFrame \\
        Serialised DataFrame per requested ticker indexed by date containing raw
        OHL(a)CV data for every date the ticker has data for.
    
    metadata.json : Metadata \\
        Serialised typed dictionary containing global static metadata for all
        stocks along with a list of all stocks and their own static metadata.
    
    download_log.json : dict[str, str] \\
        Serialised dictionary containing the outcome per requested ticker.
        The possibilities are:
        - "success": data successfully retrieved.
        - "no_data": if stock ticker is unrecognised or has no data over
        requested period.
        - "error: {error_name}": an error occurred during retrieval of requested
        ticker's data.
    """

    # Calculate end date
    end_date = pd.to_datetime(start_date) + pd.DateOffset(years=years)
    if end_date > pd.Timestamp.now():
        raise ValueError(
            f"Download error: start_date '{start_date}' and years duration "
            f"'{years} exceeds the current date."
        )

    # Prepare destination directory
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir()
    stocks_path(dest).mkdir()

    # Create metadata and log dictionaries
    metadata: Metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": "yfinance",
        "start_date": start_date,
        "end_date": (end_date - pd.DateOffset(days=1)).strftime("%Y-%m-%d"),
        "tickers": {}
    }
    download_log: dict[str, str] = {}

    for ticker in tickers:
        try:
            yf_ticker = yf.Ticker(ticker)

            df = yf_ticker.history(
                start=start_date,
                end=end_date,
                auto_adjust=False,
                actions=False,
            )

            if df.empty:
                download_log[ticker] = "no_data"
                continue

            # Normalize index
            df.index = pd.to_datetime(df.index)
            df.index = df.index.tz_localize(None).normalize()
            df.index.name = "date"

            # Rename columns to canonical names
            df = df.rename(
                columns={
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Adj Close": "adj_close",
                    "Volume": "volume",
                }
            )

            # Add adj_factor column
            df["adj_factor"] = df["adj_close"] / df["close"]
            df.loc[df["close"] == 0, "adj_factor"] = pd.NA

            # Keep only required columns
            df = df[
                [
                    "open",
                    "high",
                    "low",
                    "close",
                    "adj_close",
                    "volume",
                    "adj_factor",
                ]
            ]

            # Write parquet
            out_path = stock_path(dest, ticker)
            df.to_parquet(
                out_path,
                engine="pyarrow",
                compression="zstd",
                index=True,
            )

            # Static metadata
            info = yf_ticker.info or {}
            metadata["tickers"][ticker] = {
                "exchange": info.get("exchange"),
                "currency": info.get("currency"),
                "sector": info.get("sector").lower().replace('-',' '),
                "industry": info.get("industry").lower(),
                "shares_outstanding": info.get("sharesOutstanding"),
            }

            download_log[ticker] = "success"

        except Exception as e:
            download_log[ticker] = f"error: {type(e).__name__}"

    _atomic_write_json(metadata_path(dest), metadata)
    _atomic_write_json(log_path(dest), download_log)
