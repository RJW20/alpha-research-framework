import json
import unittest
from functools import cached_property
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

from alpha_research_framework import Universe
from alpha_research_framework.data import metadata_path, stocks_path
from alpha_research_framework.features import Features, FeatureSpec, FeatureTag
from alpha_research_framework.market_data_view import (
    MarketArray,
    MarketDataView,
)
from tests.dummy_feature import DummyFeature


class TestUniverse(unittest.TestCase):

    METADATA = {
        "start_date": "2020-01-01",
        "end_date": "2020-02-01",
        "tickers": {
            "AAA": { "shares_outstanding": 100 },
            "BBB": { "shares_outstanding": 200 }
        }
    }

    def setUp(self) -> None:
        """
        Create temporary src and path directories and write metadata and
        stock data to the src directory.
        """

        self.tmp_dir = TemporaryDirectory()
        self.src = Path(self.tmp_dir.name) / "src"
        self.src.mkdir()
        self.path = Path(self.tmp_dir.name) / "universe"

        self.shape = (len(self.dates), len(self.METADATA["tickers"]))
        self._write_metadata()
        stocks_path(self.src).mkdir(parents=True, exist_ok=True)
        for ticker in self.METADATA["tickers"]:
            self._write_stock(ticker)

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    @cached_property
    def dates(self) -> pd.DatetimeIndex:
        nyse = mcal.get_calendar('NYSE')
        schedule = nyse.schedule(
            start_date=self.METADATA["start_date"],
            end_date=self.METADATA["end_date"]
        )
        return schedule.index.astype('datetime64[ms]')

    def _write_metadata(self) -> None:
        with (metadata_path(self.src)).open("w") as f:
            json.dump(self.METADATA, f)

    def _write_stock(self, ticker: str):

        t = self.shape[0]
        df = pd.DataFrame(
            {
                "adj_close": [10.0] * t,
                "volume": [1_000] * t,
                "adj_factor": [1.0] * t,
            },
            index=self.dates,
        )
        df.to_parquet(stocks_path(self.src) / f"{ticker}.parquet")

    def _build_universe(self) -> Universe:
        return Universe(
            src=self.src,
            path=self.path,
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
            TAG = FeatureTag.PREDICTOR
            def __init__(self) -> None:
                super().__init__()
            def compute(
                self,
                market_data: MarketDataView,
                features: Features,
                out: MarketArray
            ) -> None:
                out[:] = np.ones(out.shape)

        class FeatureB(DummyFeature):
            TAG = FeatureTag.PREDICTOR
            def __init__(self) -> None:
                super().__init__()
                self._a_dependency = FeatureSpec(FeatureA, ())
                self._dependencies = {self._a_dependency}
            def compute(
                self,
                market_data: MarketDataView,
                features: Features,
                out: MarketArray
            ) -> None:
                out[:] = features[self._a_dependency][:] * 2

        universe.build_features([FeatureSpec(FeatureB, ())])

        self.assertTrue((self.path / f"{FeatureA.NAME}.dat").exists())
        feature_a = self._open_memmap(f"{FeatureA.NAME}", np.float32)
        self.assertEqual(feature_a.shape, self.shape)
        np.testing.assert_array_equal(feature_a, np.ones(self.shape))

        self.assertTrue((self.path / f"{FeatureB.NAME}.dat").exists())
        feature_2 = self._open_memmap(f"{FeatureB.NAME}", np.float32)
        self.assertEqual(feature_2.shape, self.shape)
        np.testing.assert_array_equal(feature_2, np.ones(self.shape) * 2)

    def test_cross_section_market_data(self) -> None:
        """Verify cross-section includes all market data."""

        universe = self._build_universe()
        x = universe.cross_section(0)
        self.assertListEqual(list(x.columns), ["price", "volume"])

    def test_cross_section_features(self) -> None:
        """Verify cross-section includes all predictive features."""

        universe = self._build_universe()

        class Predictor(DummyFeature):
            TAG = FeatureTag.PREDICTOR
            def __init__(self) -> None:
                super().__init__()

        class Target(DummyFeature):
            TAG = FeatureTag.TARGET
            def __init__(self) -> None:
                super().__init__()

        universe._features[FeatureSpec(Predictor, ())] = np.ones(
            self.shape,
            dtype=np.float32
        )
        universe._features[FeatureSpec(Target, ())] = np.ones(
            self.shape,
            dtype=np.float32
        )

        x = universe.cross_section(0)
        self.assertListEqual(
            list(x.columns),
            ["price", "volume", Predictor.NAME]
        )

    def test_cross_section_mask(self) -> None:
        """Verify cross-section is masked correctly."""

        universe = self._build_universe()

        for removed_stocks in range(0, self.shape[1]):
            included_stocks = self.shape[1] - removed_stocks
            universe._mask[:, :] = np.array(
                [False] * removed_stocks + [True] * included_stocks
            )

            x = universe.cross_section(0)
            self.assertEqual(len(x), included_stocks)


if __name__ == "__main__":
    unittest.main()
