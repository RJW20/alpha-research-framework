import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yfinance as yf

from alpha_research_framework.data.structure import (
    log_path,
    metadata_path,
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
    end_date: str
) -> None:
    """
    Download per-stock daily data and static metadata.

    Writes:
      - dest/stocks/{ticker}.parquet
      - dest/metadata.json
      - dest/download_log.json
    """

    # Prepare destination directory
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True, exist_ok=True)
    STOCKS_PATH = stocks_path(dest)
    STOCKS_PATH.mkdir(parents=True, exist_ok=True)
    METADATA_PATH = metadata_path(dest)
    LOG_PATH = log_path(dest)

    # Create metadata and log dictionaries
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": "yfinance",
        "start_date": start_date,
        "end_date": end_date,
        "tickers": {}
    }
    download_log = {}

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
            out_path = STOCKS_PATH / f"{ticker}.parquet"
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
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "shares_outstanding": info.get("sharesOutstanding"),
            }

            download_log[ticker] = "success"

        except Exception as e:
            download_log[ticker] = f"error: {type(e).__name__}"

    _atomic_write_json(METADATA_PATH, metadata)
    _atomic_write_json(LOG_PATH, download_log)
