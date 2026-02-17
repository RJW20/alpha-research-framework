from pathlib import Path


def stocks_path(root: Path) -> Path:
    return root / "stocks"

def stock_path(root: Path, ticker: str) -> Path:
    return stocks_path(root) / f"{ticker}.parquet"

def metadata_path(root: Path) -> Path:
    return root / "metadata.json"

def log_path(root: Path) -> Path:
    return root / "download_log.json"
