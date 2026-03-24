import unittest

from alpha_research_framework.features.derived_feature import DerivedFeature
from alpha_research_framework.features.feature import Feature


class TestDerivedFeatureSource(unittest.TestCase):

    def test_class_var_assertion(self) -> None:
        """Verify definition and type of `SOURCE` asserted when subclassing."""

        with self.assertRaises(AttributeError):
            class NoSource(DerivedFeature):
                TAG = Feature.Tag.PREDICTOR
        
        with self.assertRaises(TypeError):
            class IncompatibleSource(DerivedFeature):
                TAG = Feature.Tag.PREDICTOR
                SOURCE = str


if __name__ == "__main__":
    unittest.main()
