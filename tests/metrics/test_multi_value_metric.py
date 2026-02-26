import unittest

from alpha_research_framework.metrics import Metric, MultiValueMetric
from tests.utils import RegistryIsolatedTestCase


class TestMultiValueFeatureMeasureGroup(RegistryIsolatedTestCase):
    REGISTRY_OWNER = Metric

    def test_class_var_assertion(self) -> None:
        """
        Verify definition, type and value of `MEASURE_GROUP` asserted when
        subclassing.
        """

        with self.assertRaises(AttributeError):
            class NoMeasureGroup(MultiValueMetric):
                ID = "no_measure_group"

        with self.assertRaises(TypeError):
            class IncompatibleMeasureGroup(MultiValueMetric):
                ID = "incompatible_measure_group"
                MEASURE_GROUP = 1

        with self.assertRaises(ValueError):
            class BadMeasureGroup(MultiValueMetric):
                ID = "bad_measure_group"
                MEASURE_GROUP = ""

    
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
                MEASURE_GROUP = "measure_group"

        with self.assertRaises(TypeError):
            class IncompatibleMeasuresContainer(MultiValueMetric):
                ID = "incompatible_measures_container"
                MEASURE_GROUP = "measure_group"
                MEASURES = set()

        with self.assertRaises(TypeError):
            class IncompatibleMeasuresElement(MultiValueMetric):
                ID = "incompatible_measures_element"
                MEASURE_GROUP = "measure_group"
                MEASURES = [1]

        with self.assertRaises(ValueError):
            class BadMeasures(MultiValueMetric):
                ID = "bad_measures"
                MEASURE_GROUP = "measure_group"
                MEASURES = [""]


if __name__ == "__main__":
    unittest.main()
