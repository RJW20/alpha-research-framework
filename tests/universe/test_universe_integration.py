import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from alpha_research_framework import EquityData, Universe, Window
from alpha_research_framework.download import Metadata
from alpha_research_framework.features import (
    Feature,
    Features,
    FeatureSpec,
    FutureReturns,
)
from alpha_research_framework.market_data_view import (
    MarketArray,
    MarketDataView,
)
from tests.dummy_feature import DummyFeature
from tests.utils import create_download_dir


class TestUniverse(unittest.TestCase):

    METADATA: Metadata = {
        "created_at": "N/A",
        "source": "N/A",
        "start_date": "2020-01-01",
        "end_date": "2020-02-01",
        "tickers": {
            "AAA": {
                "exchange": "NYSE",
                "currency": "USD",
                "sector": "energy",
                "industry": "thermal coal",
                "shares_outstanding": 100
            },
            "BBB": {
                "exchange": "NYSE",
                "currency": "USD",
                "sector": "energy",
                "industry": "oil & gas drilling",
                "shares_outstanding": 100
            },
            "CCC": {
                "exchange": "NYSE",
                "currency": "USD",
                "sector": "technology",
                "industry": "solar",
                "shares_outstanding": 100
            },
        }
    }

    def setUp(self) -> None:
        """
        Create temporary src and path directories and write files to src to make
        an EquityData instance from.
        """

        self._tmp_dir = TemporaryDirectory()
        src = Path(self._tmp_dir.name) / "src"
        src.mkdir()
        self.path = Path(self._tmp_dir.name) / "universe"
        create_download_dir(src, self.METADATA)
        self._equity_data = EquityData(src)
        self.shape = (
            len(self._equity_data.dates),
            len(self._equity_data.tickers)
        )

    def tearDown(self) -> None:
        self._tmp_dir.cleanup()

    def _build_universe(self) -> Universe:
        return Universe(
            path=self.path,
            equity_data=self._equity_data,
            liquidity_threshold=0.0,
            mcap_threshold=0.0
        )
        
    def _open_memmap(self, name: str, dtype: type) -> np.memmap:
        return np.memmap(
            self.path / f"{name}.dat",
            dtype=dtype,
            mode="r",
            shape=self.shape
        )

    def test_initialisation(self) -> None:
        """Verify universe shape and creation of memmaps."""

        universe = self._build_universe()

        self.assertEqual(universe.shape, self.shape)

        self.assertTrue((self.path / "price.dat").exists())
        price = self._open_memmap("price", np.float32)
        self.assertEqual(price.shape, self.shape)

        self.assertTrue((self.path / "volume.dat").exists())
        volume = self._open_memmap("volume", np.float32)
        self.assertEqual(volume.shape, self.shape)

        self.assertTrue((self.path / "mask.dat").exists())
        mask = self._open_memmap("mask", np.bool)
        self.assertEqual(mask.shape, self.shape)
        self.assertTrue(mask.all())

    def test_build_features(self) -> None:
        """
        Verfiy features are built with intermediate dependencies and correct
        values.
        """

        universe = self._build_universe()

        class FeatureA(DummyFeature):
            def compute(
                self,
                market_data: MarketDataView,
                features: Features,
                out: MarketArray
            ) -> None:
                out[:] = np.ones(out.shape)

        class FeatureB(DummyFeature):
            __a_dependency__ = FeatureSpec(FeatureA)
            __dependencies__ = {__a_dependency__}
            def compute(
                self,
                market_data: MarketDataView,
                features: Features,
                out: MarketArray
            ) -> None:
                out[:] = features[self.__a_dependency__][:] * 2

        universe.build_features([FeatureSpec(FeatureB)])

        self.assertTrue((self.path / f"{FeatureA.__name__}.dat").exists())
        feature_a = self._open_memmap(f"{FeatureA.__name__}", np.float32)
        self.assertEqual(feature_a.shape, self.shape)
        np.testing.assert_array_equal(feature_a, np.ones(self.shape))

        self.assertTrue((self.path / f"{FeatureB.__name__}.dat").exists())
        feature_2 = self._open_memmap(f"{FeatureB.__name__}", np.float32)
        self.assertEqual(feature_2.shape, self.shape)
        np.testing.assert_array_equal(feature_2, np.ones(self.shape) * 2)

    def test_cross_section_market_data(self) -> None:
        """Verify cross-section includes all market data."""

        universe = self._build_universe()
        x = universe.cross_section(universe.dates[0])
        self.assertListEqual(list(x.keys()), ["price", "volume"])

    def test_cross_section_features(self) -> None:
        """Verify cross-section includes all predictive features."""

        universe = self._build_universe()

        class Predictor(DummyFeature):
            TAG = Feature.Tag.PREDICTOR

        class Target(DummyFeature):
            TAG = Feature.Tag.TARGET

        universe._features[FeatureSpec(Predictor)] = np.ones(
            self.shape,
            dtype=np.float32
        )
        universe._features[FeatureSpec(Target)] = np.ones(
            self.shape,
            dtype=np.float32
        )

        x = universe.cross_section(universe.dates[0])
        self.assertListEqual(
            list(x.keys()),
            ["price", "volume", Predictor.__name__]
        )

    def test_cross_section_mask(self) -> None:
        """Verify cross-section is masked correctly."""

        universe = self._build_universe()

        for removed_stocks in range(0, self.shape[1]):
            included_stocks = self.shape[1] - removed_stocks
            universe._mask[:, :] = np.array(
                [False] * removed_stocks + [True] * included_stocks
            )

            x = universe.cross_section(universe.dates[0])
            self.assertEqual(len(x["price"]), included_stocks)

    def test_future_returns_horizons(self) -> None:
        """Verify future returns includes all horizons."""

        universe = self._build_universe()

        for horizon in Window:
            universe._features[FeatureSpec(FutureReturns, horizon)] = np.ones(
                self.shape,
                dtype=np.float32
            )
        
        fut_ret = universe.future_returns(universe.dates[0])
        self.assertListEqual(list(fut_ret.keys()), list(Window))

    def test_future_returns_mask(self) -> None:
        """Verify future returns is masked correctly."""
        
        universe = self._build_universe()

        universe._features[FeatureSpec(FutureReturns, Window.DAY)] = np.ones(
                self.shape,
                dtype=np.float32
            )

        for removed_stocks in range(0, self.shape[1]):
            included_stocks = self.shape[1] - removed_stocks
            universe._mask[:, :] = np.array(
                [False] * removed_stocks + [True] * included_stocks
            )

            fut_ret = universe.future_returns(universe.dates[0])
            self.assertEqual(len(fut_ret[Window.DAY]), included_stocks)


if __name__ == "__main__":
    unittest.main()
