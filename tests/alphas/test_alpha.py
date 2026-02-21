import unittest

import numpy as np

import alpha_research_framework.market_data as md
from alpha_research_framework import Window
from alpha_research_framework.alphas import Alpha
from alpha_research_framework.alphas.alpha_error import AlphaError
from alpha_research_framework.features import FeatureSpec
from alpha_research_framework.universe import CrossSection
from tests.dummy_feature import DummyFeature


class TestAlphaSubClassValidation(unittest.TestCase):

    def test_abstract(self) -> None:
        """Verify subclass is not validated when __abstract__ is set."""

        class Dummy(Alpha):
            __abstract__ = True

    def test_name(self) -> None:
        """
        Verify definition, type and value of NAME validated when subclassing.
        """

        with self.assertRaises(AlphaError):
            class Dummy(Alpha):
                pass

        with self.assertRaises(TypeError):
            class Dummy(Alpha):
                NAME = 123

        with self.assertRaises(ValueError):
            class Dummy(Alpha):
                NAME = ""

    def test_category(self) -> None:
        """
        Verify definition, type and value of CATEGORY validated when
        subclassing.
        """

        with self.assertRaises(AlphaError):
            class Dummy(Alpha):
                NAME = "dummy"

        with self.assertRaises(TypeError):
            class Dummy(Alpha):
                NAME = "dummy"
                CATEGORY = 123

        with self.assertRaises(ValueError):
            class Dummy(Alpha):
                NAME = "dummy"
                CATEGORY = ""

    def test_horizons(self) -> None:
        """
        Verify definition and type of HORIZONS validated when subclassing.
        """

        with self.assertRaises(AlphaError):
            class Dummy(Alpha):
                NAME = "dummy"
                CATEGORY = "dummy"

        with self.assertRaises(TypeError):
            class Dummy(Alpha):
                NAME = "dummy"
                CATEGORY = "dummy"
                HORIZONS = list(Window)

        with self.assertRaises(TypeError):
            class Dummy(Alpha):
                NAME = "dummy"
                CATEGORY = "dummy"
                HORIZONS = {"a", 123}


class TestAlphaDependencies(unittest.TestCase):

    def test_required_features(self) -> None:
        """
        Verify required_features property contains all dependencies and
        FutureReturns at required horizons.
        """

        class Dummy(Alpha):
            __abstract__ = True
            HORIZONS = set(Window)
            def compute(self, x: CrossSection) -> md.Array:
                pass
            def _init_dependencies(self) -> set[FeatureSpec]:
                return {FeatureSpec(DummyFeature)}
            
        dummy = Dummy()
        self.assertEqual(len(dummy.required_features), len(Window) + 1)

    def test_compute_existence(self) -> None:
        """Verify compute method wrapper checks dependency existence."""

        class Dummy(Alpha):
            NAME = "dummy"
            CATEGORY = "dummy"
            HORIZONS = set()
            def compute(self, x: CrossSection) -> md.Array:
                pass
            def _init_dependencies(self) -> set[FeatureSpec]:
                return {FeatureSpec(DummyFeature)}

        dummy = Dummy()

        x = CrossSection()
        with self.assertRaises(AlphaError):
            dummy.compute(x)

        for feature in dummy.required_features:
            x[feature.name] = np.array([])

        dummy.compute(x)


if __name__ == "__main__":
    unittest.main()
