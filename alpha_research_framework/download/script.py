import argparse
import json
from pathlib import Path

from .download import download

# -- defaults --
DESTINATION = "data"
START_DATE = "2020-01-01"
YEARS = 1


def download_by_json() -> None:

    parser = argparse.ArgumentParser(description="Stock data downloader")
    parser.add_argument(
        "-d",
        "--destination",
        default=DESTINATION,
        type=str,
        help=(
            "Directory to write all downloaded/created files to (created or "
            "overwritten)"
        ),
    )
    parser.add_argument(
        "-t",
        "--tickers",
        type=str,
        help="json containing listing of all tickers to download data for"
    )
    parser.add_argument(
        "-s",
        "--start-date",
        default=START_DATE,
        type=str,
        help="First date to retrieve data for (yyyy-mm-dd)"
    )
    parser.add_argument(
        "-y",
        "--years",
        default=YEARS,
        type=int,
        help="Number of years to download daily data over"
    )
    args = parser.parse_args()

    with open(args.tickers, "r") as f:
        tickers = json.load(f)

    download(
        Path(args.destination),
        tickers,
        args.start_date,
        args.years
    )
