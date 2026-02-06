import unittest

from alpha_research_framework import Universe
from alpha_research_framework.features import Feature, Features, FeatureSpec
from alpha_research_framework.market_array import MarketArray
from alpha_research_framework.universe import MarketData


class Dummy(Feature):

    NAME = "dummy"

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = f"{self.NAME}_{name}"

    def compute(
        self,
        market_data: MarketData,
        features: Features,
        out: MarketArray
    ) -> None:
        pass


class FeatureA(Dummy):

    def __init__(self) -> None:
        super().__init__("a")


class FeatureB(Dummy):

    def __init__(self) -> None:
        super().__init__("b")
        self.name = self.NAME
        self.dependencies = {FeatureSpec(FeatureA, ())}


class FeatureC(Dummy):

    def __init__(self) -> None:
        super().__init__("c")
        self.name = self.NAME
        self.dependencies = {FeatureSpec(FeatureB, ())}


class TestUniverseFeatureDependencies(unittest.TestCase):

    def setUp(self) -> None:
        self.a = FeatureSpec(FeatureA, ())
        self.b = FeatureSpec(FeatureB, ())
        self.c = FeatureSpec(FeatureC, ())
        
    def test_expand_dependencies_transitive(self) -> None:
        expanded = Universe._expand_dependencies([self.c])
        self.assertEqual(expanded, {self.a, self.b, self.c})

    def test_order_dependencies(self) -> None:
        ordered = list(Universe._order_dependencies({self.b, self.c, self.a}))
        self.assertEqual(ordered, [self.a, self.b, self.c])


if __name__ == "__main__":
    unittest.main()
