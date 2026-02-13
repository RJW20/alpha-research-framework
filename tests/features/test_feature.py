import unittest

import numpy as np
import pandas as pd

from alpha_research_framework.features import (
    Feature,
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
                self._dependencies = {FeatureSpec(Predictor)}

        class PredictorOnTarget(Predictor):
            def __init__(self) -> None:
                super().__init__()
                self._dependencies = {FeatureSpec(Target)}

        class TargetOnPredictor(Target):
            def __init__(self) -> None:
                super().__init__()
                self._dependencies = {FeatureSpec(Predictor)}

        class TargetOnTarget(Target):
            def __init__(self) -> None:
                super().__init__()
                self._dependencies = {FeatureSpec(Target)}

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
                self._dependencies = {FeatureSpec(FeatureA)}

        b = FeatureB()
        features = Features()
        with self.assertRaises(FeatureError):
            b.compute({}, features, np.ndarray(shape=(10,)))

        for feature in b.dependencies:
            features[feature] = np.ndarray(shape=(10,))

        b.compute({}, features, np.ndarray(shape=(10,)))

    def test_rolling_std(self) -> None:
        """
        Verify rolling std calculation is same as pandas built in dataframe
        version.
        """

        rng = np.random.default_rng(0)
        x = rng.uniform(0, 10, (10000, 10)).astype(np.float32)
        nan_mask = rng.uniform(size=x.shape) < 0.1
        x[nan_mask] = np.nan
        for lookback in [1, 10, 100, 1000]:
            out = np.empty_like(x, dtype=np.float32)
            Feature._rolling_std(x, lookback, out)
            df = pd.DataFrame(x)
            expected = df.rolling(lookback, min_periods=1).std()
            np.testing.assert_array_almost_equal(
                out,
                expected.to_numpy(dtype=np.float32),
                decimal=5
            )


if __name__ == "__main__":
    unittest.main()
