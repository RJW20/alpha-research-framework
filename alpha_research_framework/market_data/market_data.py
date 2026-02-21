from typing import Protocol

from alpha_research_framework.market_data.array import Array


class MarketData(Protocol):
    """View over time * stock market data."""

    @property
    def price(self) -> Array:
        ...

    @property
    def volume(self) -> Array:
        ...
