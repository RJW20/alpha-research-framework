import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, override

import numpy as np
import pandas as pd

import alpha_research_framework.features as features
import alpha_research_framework.observables as observables
import alpha_research_framework.universe.build as build
from alpha_research_framework.scalar import Scalar
from alpha_research_framework.window import Window


class TestBuildFeatureObservableDeduction(unittest.TestCase):

    def test_deduce_cross_sectional_features(self) -> None:
        """
        Verify deduced features is the union of all alphas `REQUIRED_FEATURES.
        """

        class AlphaA:
            REQUIRED_FEATURES = {1, 2, 3}

        class AlphaB:
            REQUIRED_FEATURES = {3, 4, 5}

        self.assertSetEqual(
            build._deduce_cross_sectional_features([AlphaA, AlphaB]),
            {1, 2, 3, 4, 5},
        )

    def test_scan_feature_tree(self) -> None:
        """Verify all observable heads and feature nodes are returned."""

        class HeadA(observables.Observable):
            NAME = "head_a"
        class PrimitiveA(features.PrimitiveFeature):
            TAG = features.Feature.Tag.PREDICTOR
            OBSERVABLE = HeadA
        class DerivedA(features.DerivedFeature):
            TAG = features.Feature.Tag.PREDICTOR
            SOURCE = PrimitiveA

        class HeadB(observables.Observable):
            NAME = "head_b"
        class PrimitiveB(features.PrimitiveFeature):
            TAG = features.Feature.Tag.PREDICTOR
            OBSERVABLE = HeadB

        observables_, features_ = build._scan_feature_tree(
            [DerivedA, PrimitiveB],
        )
        self.assertSetEqual(observables_, {HeadA, HeadB})
        self.assertSetEqual(features_, {PrimitiveA, DerivedA, PrimitiveB})


class TestBuildMarketDataMask(unittest.TestCase):

    def test_market_data(self) -> None:
        """
        Verify `market_data` is filled with values for every observable it
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


class TestBuildFeatures(unittest.TestCase):

    def test_order(self) -> None:
        """Verify a feature's `SOURCE` always comes before itself."""

        class Dummy(observables.Observable):
            NAME = "dummy"

        class Primitive(features.PrimitiveFeature):
            TAG = features.Feature.Tag.PREDICTOR
            OBSERVABLE = Dummy

        class Derived(features.DerivedFeature):
            TAG = features.Feature.Tag.PREDICTOR
            SOURCE = Primitive

        self.assertListEqual(
            list(build._order([Primitive, Derived])),
            [Primitive, Derived],
        )
        self.assertListEqual(
            list(build._order([Derived, Primitive])),
            [Primitive, Derived],
        )

    def test_build(self) -> None:
        """
        Verify all requested features (primitive and derived) and built with
        correct values.
        """

        T, N = 1000, 100

        class OnesObservable(observables.Observable):
            NAME = "ones"

        class OnesFeature(features.PrimitiveFeature):
            TAG = features.Feature.Tag.PREDICTOR
            OBSERVABLE = OnesObservable

        class Double(features.transforms.Transform):
            @classmethod
            @override
            def compute(cls, x: list[int], **kwargs: Any) -> None:
                x[:] = [i * 2 for i in x]

        TwosFeature = Double(OnesFeature)

        tmp_dir = TemporaryDirectory()
        features_ = build._allocate_storage(
            [OnesFeature, TwosFeature],
            path=Path(tmp_dir.name),
            shape=(T, N),
        )

        market_data = {OnesObservable: np.ones((T,N))}

        build._build(features_, market_data=market_data)

        np.testing.assert_array_equal(features_[OnesFeature], np.ones((T,N)))
        np.testing.assert_array_equal(features_[TwosFeature], np.full((T,N), 2))

        tmp_dir.cleanup()


if __name__ == "__main__":
    unittest.main()
