import unittest

import numpy as np

from alpha_research_framework.features import (
    FeatureError,
    Features,
    FeatureSpec,
    FeatureTag,
)
from tests.dummy_feature import DummyFeature


class TestFeature(unittest.TestCase):

    def test_dependencies_tag(self) -> None:
        """Verify __init__ method wrapper checks dependency TAGs."""

        class Predictor(DummyFeature):
            TAG = FeatureTag.PREDICTOR
            def __init__(self) -> None:
                super().__init__()

        class Target(DummyFeature):
            TAG = FeatureTag.TARGET
            def __init__(self) -> None:
                super().__init__()

        class PredictorOnPredictor(Predictor):
            def __init__(self) -> None:
                super().__init__()
                self._dependencies = {FeatureSpec(Predictor, ())}

        class PredictorOnTarget(Predictor):
            def __init__(self) -> None:
                super().__init__()
                self._dependencies = {FeatureSpec(Target, ())}

        class TargetOnPredictor(Target):
            def __init__(self) -> None:
                super().__init__()
                self._dependencies = {FeatureSpec(Predictor, ())}

        class TargetOnTarget(Target):
            def __init__(self) -> None:
                super().__init__()
                self._dependencies = {FeatureSpec(Target, ())}

        feature = PredictorOnPredictor()
        with self.assertRaises(FeatureError):
            feature = PredictorOnTarget()
        feature = TargetOnPredictor()
        feature = TargetOnTarget()

    def test_dependencies_existence(self) -> None:
        """Verify compute method wrapper checks dependency existence."""

        class FeatureA(DummyFeature):
            TAG = FeatureTag.PREDICTOR
            def __init__(self) -> None:
                super().__init__()

        class FeatureB(DummyFeature):
            TAG = FeatureTag.PREDICTOR
            def __init__(self) -> None:
                super().__init__()
                self._dependencies = {FeatureSpec(FeatureA, ())}

        b = FeatureB()
        features = Features()
        with self.assertRaises(FeatureError):
            b.compute({}, features, np.ndarray(shape=(10,)))

        for feature in b.dependencies:
            features[feature] = np.ndarray(shape=(10,))

        b.compute({}, features, np.ndarray(shape=(10,)))


if __name__ == "__main__":
    unittest.main()
