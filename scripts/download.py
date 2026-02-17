import json
from pathlib import Path

from alpha_research_framework import download as dl

DEST = Path("data")
START_DATE = "2000-01-01"
END_DATE = "2020-01-01"


def download() -> None:
    with open("tickers/sp500.json", "r") as f:
        tickers = json.load(f)
    dl(DEST, tickers, START_DATE, END_DATE)


if __name__ == "__main__":
    download()
