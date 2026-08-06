import unittest

import alpha_research_framework.series as series
from alpha_research_framework.signals.series_signal import SeriesSignal


class TestSeriesSignalSeries(unittest.TestCase):

    def test_class_var_assertion(self) -> None:
        """
        Verify definition and type of `SERIES` asserted when subclassing.
        """

        with self.assertRaises(AttributeError):
            class NoSeries(SeriesSignal):
                pass

        with self.assertRaises(TypeError):
            class IncompatibleSeries(SeriesSignal):
                SERIES = str


class TestSeriesSignalCompute(unittest.TestCase):

    def test_absent_series(self) -> None:
        """
        Verify a `ValueError` is thrown only if the cross-section doesn't
        contain `SERIES`.
        """

        class DummySeries(series.Series, abstract=True):
            pass

        class DummySignal(SeriesSignal):
            SERIES = DummySeries

        DummySignal.compute({DummySeries: []}, {})
        with self.assertRaises(ValueError):
            DummySignal.compute({}, {})

    def test_extracted_values(self) -> None:
        """Verify values extracted from cross-section pertain to `SERIES`."""

        class DummySeries(series.Series, abstract=True):
            pass

        class DummySignal(SeriesSignal):
            SERIES = DummySeries

        dummy_values = ["a", "b"]
        cross_section = {DummySeries: dummy_values}
        self.assertListEqual(
            DummySignal.compute(cross_section, {}),
            dummy_values,
        )


if __name__ == "__main__":
    unittest.main()
