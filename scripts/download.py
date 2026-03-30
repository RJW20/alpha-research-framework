import argparse
import json
from pathlib import Path

import alpha_research_framework as arf

# -- defaults --
DESTINATION = "data"
TICKERS = "tickers/sp500.json"
START_DATE = "2020-01-01"
YEARS = 1


def download() -> None:

    parser = argparse.ArgumentParser(description="Stock data downloader")
    parser.add_argument(
        "-d",
        "--destination",
        default=DESTINATION,
        type=str,
        help="folder stock data is saved to (created or overwritten)"
    )
    parser.add_argument(
        "-t",
        "--tickers",
        default=TICKERS,
        type=str,
        help="json containing tickers to download data for"
    )
    parser.add_argument(
        "-s",
        "--start-date",
        default=START_DATE,
        type=str,
        help="date to start download from"
    )
    parser.add_argument(
        "-y",
        "--years",
        default=YEARS,
        type=int,
        help="number of years to download data for"
    )
    args = parser.parse_args()

    with open(args.tickers, "r") as f:
        tickers = json.load(f)

    arf.download(
        Path(args.destination),
        tickers,
        args.start_date,
        args.years
    )


if __name__ == "__main__":
    download()
