from dataclasses import dataclass

import alpha_research_framework.market_data as md


@dataclass
class DummyMarketData:
    price: md.Array
    volume: md.Array
