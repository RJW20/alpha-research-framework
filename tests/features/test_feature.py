import unittest

import numpy as np

from alpha_research_framework.features import Feature, Features, FeatureSpec
from alpha_research_framework.market_array import MarketArray
from alpha_research_framework.universe import MarketData


class Dummy(Feature):
    """Dummy feature dependent on itself."""

    NAME = "dummy"

    def __init__(self) -> None:
        super().__init__()
        self.name = f"{self.NAME}"
        self.dependencies = {FeatureSpec(Dummy, ())}
        
    def compute(
        self,
        market_data: MarketData,
        features: Features,
        out: MarketArray
    ) -> None:
        pass


class TestFeature(unittest.TestCase):

    def test_dependencies_check(self) -> None:
        """Confirm compute method wrapper checks dependency existence."""

        dummy = Dummy()
        features = Features()
        with self.assertRaises(ValueError):
            dummy.compute({}, features, np.ndarray(shape=(10,)))

        for feature_spec in dummy.dependencies:
            features[feature_spec] = np.ndarray(shape=(10,))

        dummy.compute({}, features, np.ndarray(shape=(10,)))


if __name__ == "__main__":
    unittest.main()
