from alpha_research_framework.features import Feature, Features
from alpha_research_framework.market_array import MarketArray
from alpha_research_framework.universe import MarketData


class DummyFeature(Feature):

    NAME: str = "dummy"

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = f"{self.NAME}_{name.lower()}"

    def compute(
        self,
        market_data: MarketData,
        features: Features,
        out: MarketArray
    ) -> None:
        pass
