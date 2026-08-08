import unittest

import numpy as np
import pandas as pd

import alpha_research_framework.observables as observables
import alpha_research_framework.series as series
import alpha_research_framework.signals as signals
import alpha_research_framework.universe.build as build
from alpha_research_framework.scalar import Scalar
from alpha_research_framework.series.combined_series import CombinedSeries
from alpha_research_framework.series.observable_series import ObservableSeries
from alpha_research_framework.series.transformed_series import TransformedSeries
from alpha_research_framework.signals.combined_signal import CombinedSignal
from alpha_research_framework.signals.negated_signal import NegatedSignal
from alpha_research_framework.signals.series_signal import SeriesSignal
from alpha_research_framework.window import Window

# ------------------------------------------------------------------------------
# Series/Observable deduction
# ------------------------------------------------------------------------------

class TestBuildSeriesDeduction(unittest.TestCase):
    """Verify all roots of `Signal` DAGs are returned."""

    @staticmethod
    def _new_series(name: str) -> type[series.Series]:
        return type(
            name,
            (series.Series,),
            {
                "TAG": series.Series.Tag.PREDICTOR
            }
        )

    @staticmethod
    def _new_series_signal(
        name: str,
        series: type[series.Series],
    ) -> type[SeriesSignal]:
        return type(
            name,
            (SeriesSignal,),
            {
                "SERIES": series,
            }
        )

    @staticmethod
    def _new_combined_signal(
        name: str,
        source_left: type[signals.Signal],
        source_right: type[signals.Signal],
    ) -> type[CombinedSignal]:
        return type(
            name,
            (CombinedSignal,),
            {
                "SOURCE_LEFT": source_left,
                "SOURCE_RIGHT": source_right,
                "BINARY_OP": lambda x,y: x + y
            }
        )

    @staticmethod
    def _new_negated_signal(
        name: str,
        source: type[signals.Signal],
    ) -> type[NegatedSignal]:
        return type(
            name,
            (NegatedSignal,),
            {
                "SOURCE": source,
            }
        )

    def test_simple_chain(self) -> None:
        """
        Graph:
            A
            |
            v
            B
            |
            v
            C
        Expected roots: A
        """

        A = self._new_series("A")
        B = self._new_series_signal("B", A)
        C = self._new_negated_signal("C", B)
        self.assertSetEqual(
            build._deduce_cross_sectional_series([C]),
            {A},
        )

    def test_merge(self) -> None:
        r"""
        Graph:
          A   B
          |   |
          v   v
          C   D
           \ /
            v
            E
        Expected roots: A, B
        """

        A = self._new_series("A")
        B = self._new_series("B")
        C = self._new_series_signal("C", A)
        D = self._new_series_signal("D", B)
        E = self._new_combined_signal("C", C, D)
        self.assertSetEqual(
            build._deduce_cross_sectional_series([E]),
            {A, B},
        )

    def test_diamond(self) -> None:
        r"""
        Graph:
            A
           / \
          v   v
          B   C
           \ /
            v
            D
        Expected roots: A
        """

        A = self._new_series("A")
        B = self._new_series_signal("B", A)
        C = self._new_series_signal("C", A)
        D = self._new_combined_signal("D", B, C)
        self.assertSetEqual(
            build._deduce_cross_sectional_series([D]),
            {A},
        )

    def test_large(self) -> None:
        r"""
        Graph:
            A     B   C
           / \    |   |
          v   v   v   v
          D   E   F   G
           \ / \ / \ /
            v   v   v
            H   I   J
             \ /
              K
        Expected roots: A, B, C
        """

        A = self._new_series("A")
        B = self._new_series("B")
        C = self._new_series("C")
        D = self._new_series_signal("D", A)
        E = self._new_series_signal("E", A)
        F = self._new_series_signal("F", B)
        G = self._new_series_signal("G", C)
        H = self._new_combined_signal("H", D, E)
        I = self._new_combined_signal("I", E, F)
        J = self._new_combined_signal("J", F, G)
        K = self._new_combined_signal("K", H, I)
        self.assertSetEqual(
            build._deduce_cross_sectional_series([K, J]),
            {A, B, C},
        )


class TestBuildObservablesDeduction(unittest.TestCase):
    """Verify all roots of `Series` DAGs are returned."""

    @staticmethod
    def _new_observable(name: str) -> type[observables.Observable]:
        return type(
            name,
            (observables.Observable,),
            {
                "NAME": name
            }
        )

    @staticmethod
    def _new_observable_series(
        name: str,
        observable: type[observables.Observable],
    ) -> type[ObservableSeries]:
        return type(
            name,
            (ObservableSeries,),
            {
                "OBSERVABLE": observable,
                "TAG": series.Series.Tag.PREDICTOR,
            }
        )

    @staticmethod
    def _new_combined_series(
        name: str,
        source_left: type[series.Series],
        source_right: type[series.Series],
    ) -> type[CombinedSeries]:
        return type(
            name,
            (CombinedSeries,),
            {
                "TAG": series.Series.Tag.PREDICTOR,
                "SOURCE_LEFT": source_left,
                "SOURCE_RIGHT": source_right,
                "BINARY_OP": lambda x,y: x + y
            }
        )

    @staticmethod
    def _new_transformed_series(
        name: str,
        source: type[series.Series],
    ) -> type[TransformedSeries]:
        return type(
            name,
            (TransformedSeries,),
            {
                "TAG": series.Series.Tag.PREDICTOR,
                "SOURCE": source,
                "TRANSFORM": lambda x: x
            }
        )

    def test_simple_chain(self) -> None:
        """
        Graph:
            A
            |
            v
            B
            |
            v
            C
        Expected roots: A
        """

        A = self._new_observable("A")
        B = self._new_observable_series("B", A)
        C = self._new_transformed_series("C", B)
        self.assertSetEqual(
            build._deduce_market_data_observables([C]),
            {A},
        )

    def test_merge(self) -> None:
        r"""
        Graph:
          A   B
          |   |
          v   v
          C   D
           \ /
            v
            E
        Expected roots: A, B
        """

        A = self._new_observable("A")
        B = self._new_observable("B")
        C = self._new_observable_series("C", A)
        D = self._new_observable_series("D", B)
        E = self._new_combined_series("C", C, D)
        self.assertSetEqual(
            build._deduce_market_data_observables([E]),
            {A, B},
        )

    def test_diamond(self) -> None:
        r"""
        Graph:
            A
           / \
          v   v
          B   C
           \ /
            v
            D
        Expected roots: A
        """

        A = self._new_observable("A")
        B = self._new_observable_series("B", A)
        C = self._new_observable_series("C", A)
        D = self._new_combined_series("D", B, C)
        self.assertSetEqual(
            build._deduce_market_data_observables([D]),
            {A},
        )

    def test_large(self) -> None:
        r"""
        Graph:
            A     B   C
           / \    |   |
          v   v   v   v
          D   E   F   G
           \ / \ / \ /
            v   v   v
            H   I   J
             \ /
              K
        Expected roots: A, B, C
        """

        A = self._new_observable("A")
        B = self._new_observable("B")
        C = self._new_observable("C")
        D = self._new_observable_series("D", A)
        E = self._new_observable_series("E", A)
        F = self._new_observable_series("F", B)
        G = self._new_observable_series("G", C)
        H = self._new_combined_series("H", D, E)
        I = self._new_combined_series("I", E, F)
        J = self._new_combined_series("J", F, G)
        K = self._new_combined_series("K", H, I)
        self.assertSetEqual(
            build._deduce_market_data_observables([K, J]),
            {A, B, C},
        )

# ------------------------------------------------------------------------------
# Market Data and Mask building
# ------------------------------------------------------------------------------


class TestBuildMarketDataMask(unittest.TestCase):

    def test_market_data(self) -> None:
        """
        Verify `market_data` is filled with values for every `Observable` it
        contains.
        """

        T, N = 100, 10

        observables_: list[type[observables.Observable]] = [
            observables.Open,
            observables.High,
            observables.Low,
            observables.Close,
            observables.Volume,
        ]

        market_data = {
            obs: np.empty(shape=(T, N), dtype=Scalar) for obs in observables_
        }

        rng = np.random.default_rng()
        column = rng.integers(0, N)
        ticker_data = pd.DataFrame(
            {
                obs.NAME: rng.uniform(size=T) for obs in observables_
            }
        )

        build._populate_market_data(market_data, column, ticker_data)

        for obs in observables_:
            np.testing.assert_array_equal(
                market_data[obs][:, column],
                ticker_data[obs.NAME]
            )

    def test_mask_all_conditions_met(self) -> None:
        """Verify a ticker that meets all requirements is never masked out."""

        mask = np.empty(shape=(3,1), dtype=bool)

        ticker_info = {"shares_outstanding": 1_000_000}
        ticker_data = pd.DataFrame(
            {
                "adj_close": [10.0, np.nan, 10.0],
                "volume": [1_000, 1_000, 1_000],
            }
        )

        build._populate_mask(
            mask,
            0,
            ticker_info,
            ticker_data,
            liquidity_threshold=5_000,
            mcap_threshold=5_000_000,
            lookback=Window.WEEK,
        )

        self.assertTrue(mask.all())

    def test_mask_nan_adj_close(self) -> None:
        """
        Verify a ticker with no non NaN `adj_close` over `lookback` is masked
        out.
        """

        mask = np.empty(shape=(3,1), dtype=bool)

        ticker_info = {"shares_outstanding": 1_000_000}
        ticker_data = pd.DataFrame(
            {
                "adj_close": [10.0, np.nan, 10.0],
                "volume": [1_000, 1_000, 1_000],
            }
        )

        build._populate_mask(
            mask,
            0,
            ticker_info,
            ticker_data,
            liquidity_threshold=0,
            mcap_threshold=0,
            lookback=Window.DAY,
        )

        np.testing.assert_array_equal(
            mask,
            np.array([[True], [False], [True]]),
        )

    def test_mask_liquidity_threshold(self) -> None:
        """
        Verify a ticker that falls below the average liquidity requirement over
        `lookback` is masked out.
        """

        mask = np.empty(shape=(3,1), dtype=bool)

        ticker_info = {"shares_outstanding": 1_000_000}
        ticker_data = pd.DataFrame(
            {
                "adj_close": [10.0, 10.0, 10.0],
                "volume": [10, 100, 1_000],
            }
        )

        build._populate_mask(
            mask,
            0,
            ticker_info,
            ticker_data,
            liquidity_threshold=1_000,
            mcap_threshold=0,
            lookback=Window.DAY,
        )

        np.testing.assert_array_equal(
            mask,
            np.array([[False], [True], [True]]),
        )

    def test_mask_mcap_threshold(self) -> None:
        """
        Verify a ticker that falls below the market cap requirement is masked
        out.
        """

        mask = np.empty(shape=(3,1), dtype=bool)

        ticker_info = {"shares_outstanding": 1_000_000}
        ticker_data = pd.DataFrame(
            {
                "adj_close": [1.0, 10.0, 100.0],
                "volume": [1_000, 1_000, 1_000],
            }
        )

        build._populate_mask(
            mask,
            0,
            ticker_info,
            ticker_data,
            liquidity_threshold=1_000,
            mcap_threshold=10_000_000,
            lookback=Window.DAY,
        )

        np.testing.assert_array_equal(
            mask,
            np.array([[False], [True], [True]]),
        )


if __name__ == "__main__":
    unittest.main()
