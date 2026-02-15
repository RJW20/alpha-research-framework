import unittest

from alpha_research_framework import Universe
from alpha_research_framework.features import FeatureSpec
from tests.dummy_feature import DummyFeature


class FeatureA(DummyFeature):
    pass


class FeatureB(DummyFeature):
    __dependencies__ = {FeatureSpec(FeatureA)}


class FeatureC(DummyFeature):
    __dependencies__ = {FeatureSpec(FeatureB)}


class TestUniverseFeatureDependencies(unittest.TestCase):

    def setUp(self) -> None:
        self.a = FeatureSpec(FeatureA)
        self.b = FeatureSpec(FeatureB)
        self.c = FeatureSpec(FeatureC)
        
    def test_expand_dependencies_transitive(self) -> None:
        expanded = Universe._expand_dependencies([self.c])
        self.assertEqual(expanded, {self.a, self.b, self.c})

    def test_order_dependencies(self) -> None:
        ordered = list(Universe._order_dependencies({self.b, self.c, self.a}))
        self.assertEqual(ordered, [self.a, self.b, self.c])


if __name__ == "__main__":
    unittest.main()
