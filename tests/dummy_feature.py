from alpha_research_framework.features import Feature, Features
from alpha_research_framework.market_array import MarketArray
from alpha_research_framework.universe import MarketData


class DummyFeature(Feature):

    NAME: str = "dummy"

    def __init__(self) -> None:
        super().__init__()
        self.name = f"{self.NAME}"

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        cls.NAME = cls.NAME + "_" + cls.__name__.lower()

    def compute(
        self,
        market_data: MarketData,
        features: Features,
        out: MarketArray
    ) -> None:
        pass
