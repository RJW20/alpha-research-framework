import unittest

from alpha_research_framework import Universe
from alpha_research_framework.features import Feature


class FeatureA(Feature, abstract=True):
    DEPENDENCIES = set()

class FeatureB(Feature, abstract=True):
    DEPENDENCIES = {FeatureA}

class FeatureC(Feature, abstract=True):
    DEPENDENCIES = {FeatureB}

class TestUniverseFeatureDependencies(unittest.TestCase):

    def test_expand_dependencies_transitive(self) -> None:
        """Verify `_expand_dependencies` is transitive."""

        expanded = Universe._expand_dependencies([FeatureC])
        self.assertSetEqual(expanded, {FeatureA, FeatureB, FeatureC})

    def test_order_dependencies(self) -> None:
        """
        Verify `_order_dependencies` returns a feature's dependencies before
        itself.
        """

        ordered = list(
            Universe._order_dependencies([FeatureB, FeatureC, FeatureA])
        )
        self.assertListEqual(ordered, [FeatureA, FeatureB, FeatureC])


if __name__ == "__main__":
    unittest.main()
