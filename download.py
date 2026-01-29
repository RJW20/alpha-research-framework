from pathlib import Path

from alpha_research_framework import download as dl

DEST = Path("data")
TICKERS = ['AAPL', 'BMX', 'SNOW', 'ZN']
START_DATE = "2020-01-01"
END_DATE = "2021-01-01"


def download() -> None:
    dl(DEST, TICKERS, START_DATE, END_DATE)


if __name__ == "__main__":
    download()
