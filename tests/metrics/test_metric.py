import unittest

from alpha_research_framework.metrics import Metric
from alpha_research_framework.metrics.metric_error import MetricError
from tests.utils import RegistryIsolatedTestCase


class TestMetricSubClassValidation(RegistryIsolatedTestCase):
    REGISTRY_OWNER = Metric

    def test_id(self) -> None:
        """
        Verify definition, type and value of ID validated when subclassing.
        """

        with self.assertRaises(MetricError):
            class Dummy(Metric):
                pass

        with self.assertRaises(TypeError):
            class Dummy(Metric):
                ID = 1

        with self.assertRaises(ValueError):
            class Dummy(Metric):
                ID = ""

        class Dummy(Metric):
            ID = "dummy"
        with self.assertRaises(MetricError):
            class Dummy(Metric):
                ID = "dummy"

    def test_from_id(self) -> None:
        """Verify metric returned has correct id."""

        class Dummy(Metric):
            ID = "dummy"
        self.assertEqual(Metric.from_id(Dummy.ID).ID, Dummy.ID)

    def test_from_invalid_id(self) -> None:
        """Verify an error is raised when metric id is fictional."""

        with self.assertRaises(MetricError):
            Metric.from_id("abc")


if __name__ == "__main__":
    unittest.main()
