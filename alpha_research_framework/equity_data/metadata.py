from typing import TypedDict


class StockInfo(TypedDict):
    """Static stock information."""

    exchange: str
    currency: str
    sector: str
    industry: str
    shares_outstanding: int


class Metadata(TypedDict):
    """Equity data metadata."""
    
    created_at: str
    source: str
    start_date: str
    end_date: str
    tickers: dict[str, StockInfo]
