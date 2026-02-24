import unittest
from typing import override

import numpy as np
import pandas as pd

import alpha_research_framework.market_data as md
from alpha_research_framework.features.feature import (
    DependencyError,
    Feature,
    Features,
)
from tests.utils import RegistryIsolatedTestCase


class TestFeatureTag(RegistryIsolatedTestCase):
    REGISTRY_OWNER = Feature

    def test_class_var_assertion(self) -> None:
        """Verify definition and type of `TAG` asserted when subclassing."""

        with self.assertRaises(AttributeError):
            class NoTag(Feature):
                ID = "no_tag"
                pass

        with self.assertRaises(TypeError):
            class IncompatibleTag(Feature):
                ID = "incompatible_tag"
                TAG = 1


class TestFeatureDependencies(RegistryIsolatedTestCase):
    REGISTRY_OWNER = Feature

    def test_class_var_assertion(self) -> None:
        """
        Verify definition and type of `DEPENDENCIES` asserted when subclassing.
        """

        with self.assertRaises(AttributeError):
            class NoDependencies(Feature):
                ID = "no_dependencies"
                TAG = Feature.Tag.PREDICTOR

        with self.assertRaises(TypeError):
            class IncompatibleDependenciesContainer(Feature):
                ID = "incompatible_dependencies_container"
                TAG = Feature.Tag.PREDICTOR
                DEPENDENCIES = list()

        with self.assertRaises(TypeError):
            class IncompatibleDependenciesElement(Feature):
                ID = "incompatible_dependencies_element"
                TAG = Feature.Tag.PREDICTOR
                DEPENDENCIES = {1}

    def test_assert_tags(self) -> None:
        """
        Verify `_assert_dependencies_tags` asserts `Feature.TAG < cls.TAG` for
        all features in `cls.DEPENDENCIES`.
        """

        class Predictor(Feature, abstract=True):
            TAG = Feature.Tag.PREDICTOR

        class Target(Feature, abstract=True):
            TAG = Feature.Tag.TARGET

        class PredictorOnPredictor(Feature, abstract=True):
            TAG = Feature.Tag.PREDICTOR
            DEPENDENCIES = {Predictor}
        PredictorOnPredictor._assert_dependencies_tags()

        class PredictorOnTarget(Feature, abstract=True):
            TAG = Feature.Tag.PREDICTOR
            DEPENDENCIES = {Target}
        with self.assertRaises(ValueError):
            PredictorOnTarget._assert_dependencies_tags()

        class TargetOnPredictor(Feature, abstract=True):
            TAG = Feature.Tag.TARGET
            DEPENDENCIES = {Predictor}
        TargetOnPredictor._assert_dependencies_tags()

        class TargetOnTarget(Feature, abstract=True):
            TAG = Feature.Tag.TARGET
            DEPENDENCIES = {Target}
        TargetOnTarget._assert_dependencies_tags()


    def test_wrap_compute(self) -> None:
        """
        Verify `compute` is wrapped wtih custom error reporting for incomplete
        `features` argument.
        """

        class DependedOn(Feature):
            ID = "depended_on"
            TAG = Feature.Tag.PREDICTOR
            DEPENDENCIES = set()
    
        class HasCompute(Feature, abstract=True):
            DEPENDENCIES = {DependedOn}
            @classmethod
            @override
            def compute(
                cls,
                market_data: md.MarketData,
                features: Features,
                out: md.Array,
            ) -> None:
                # attempt to use the dependency
                features[DependedOn.ID]

        HasCompute._wrap_compute()

        with self.assertRaises(DependencyError):
            HasCompute.compute(None, dict(), None)

        HasCompute.compute(None, {DependedOn.ID: None}, None)


class TestFeatureCalculations(unittest.TestCase):

    def test_rolling_std(self) -> None:
        """
        Verify `_rolling_std` calculation is same as `pandas` built in
        `DataFrame` version.
        """

        rng = np.random.default_rng(0)
        x = rng.uniform(0, 10, (10000, 10)).astype(md.Scalar)
        nan_mask = rng.uniform(size=x.shape) < 0.1
        x[nan_mask] = np.nan
        for lookback in [1, 10, 100, 1000]:
            out = np.empty_like(x, dtype=md.Scalar)
            Feature._rolling_std(x, lookback, out)
            df = pd.DataFrame(x)
            expected = df.rolling(lookback, min_periods=1).std()
            np.testing.assert_array_almost_equal(
                out,
                expected.to_numpy(dtype=md.Scalar),
                decimal=5
            )


if __name__ == "__main__":
    unittest.main()
