import unittest

from alpha_research_framework.metrics import Metric, MultiValueMetric
from tests.utils import RegistryIsolatedTestCase

    
class TestMultiValueFeatureMeasures(RegistryIsolatedTestCase):
    REGISTRY_OWNER = Metric

    def test_class_var_assertion(self) -> None:
        """
        Verify definition, type and value of `MEASURES` asserted when
        subclassing.
        """

        with self.assertRaises(AttributeError):
            class NoMeasures(MultiValueMetric):
                ID = "no_measure"

        with self.assertRaises(TypeError):
            class IncompatibleMeasuresContainer(MultiValueMetric):
                ID = "incompatible_measures_container"
                MEASURES = set()

        with self.assertRaises(TypeError):
            class IncompatibleMeasuresElement(MultiValueMetric):
                ID = "incompatible_measures_element"
                MEASURES = [1]

        with self.assertRaises(ValueError):
            class BadMeasures(MultiValueMetric):
                ID = "bad_measures"
                MEASURES = [""]


if __name__ == "__main__":
    unittest.main()
