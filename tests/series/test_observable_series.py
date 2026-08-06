import unittest

import alpha_research_framework.observables as observables
from alpha_research_framework.series.observable_series import ObservableSeries
from alpha_research_framework.series.series import Series


class TestObservableSeriesObservable(unittest.TestCase):
     
    def test_class_var_assertion(self) -> None:
        """
        Verify definiition and type of `OBSERVABLE` asserted when subclassing.
        """

        with self.assertRaises(AttributeError):
            class NoObservable(ObservableSeries):
                TAG = Series.Tag.PREDICTOR
        
        with self.assertRaises(TypeError):
            class IncompatibleObservable(ObservableSeries):
                TAG = Series.Tag.PREDICTOR
                OBSERVABLE = str


class TestObservableSeriesCompute(unittest.TestCase):

    def test_absent_observable(self) -> None:
        """
        Verify a `ValueError` is thrown only if the market data doesn't contain
        `OBSERVABLE`.
        """

        class DummyObservable(observables.Observable):
            NAME = "dummy"

        class DummySeries(ObservableSeries):
            TAG = Series.Tag.PREDICTOR
            OBSERVABLE = DummyObservable

        DummySeries.compute({DummyObservable: []}, {}, None)
        with self.assertRaises(ValueError):
            DummySeries.compute({}, {}, None)

    def test_extracted_values(self) -> None:
        """Verify values extracted from market pertain to `OBSERVABLE`."""

        class DummyObservable(observables.Observable):
            NAME = "dummy"

        class DummySeries(ObservableSeries):
            TAG = Series.Tag.PREDICTOR
            OBSERVABLE = DummyObservable

        dummy_values = ["a", "b"]
        market_data = {DummyObservable: dummy_values}
        self.assertListEqual(
            DummySeries.compute(market_data, {}, None),
            dummy_values,
        )


if __name__ == "__main__":
    unittest.main()
