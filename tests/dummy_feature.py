from alpha_research_framework.features import Feature, Features
from alpha_research_framework.market_data_view import (
    MarketArray,
    MarketDataView,
)


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
        market_data: MarketDataView,
        features: Features,
        out: MarketArray
    ) -> None:
        pass
