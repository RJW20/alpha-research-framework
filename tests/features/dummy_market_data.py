from dataclasses import dataclass

from alpha_research_framework.market_data_view import MarketArray


@dataclass
class DummyMarketData:
    price: MarketArray
    volume: MarketArray
