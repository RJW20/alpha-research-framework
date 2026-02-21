import unittest

from alpha_research_framework.metrics import Metric
from alpha_research_framework.metrics.metric_error import MetricError


class TestMetricSubClassValidation(unittest.TestCase):

    def tearDown(self) -> None:
        Metric.__registry__.clear()

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

        class Dummy1(Metric):
            ID = "dummy"
        with self.assertRaises(MetricError):
            class Dummy2(Metric):
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
