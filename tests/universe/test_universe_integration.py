import json
import unittest
from functools import cached_property
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from alpha_research_framework import Universe
from alpha_research_framework.data import metadata_path, stocks_path


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
        return pd.date_range(
            self.METADATA["start_date"],
            self.METADATA["end_date"],
            freq='B'
        ).astype('datetime64[ms]')

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

    def _open_memmap(self, name: str, dtype: type) -> np.memmap:
        return np.memmap(
            self.path / f"{name}.dat",
            dtype=dtype,
            mode="r",
            shape=self.shape
        )

    def test_initialisation(self) -> None:
        """Verify universe shape and creation of memmaps."""

        u = Universe(
            src=self.src,
            path=self.path,
            liquidity_threshold=0.0,
            mcap_threshold=0.0
        )
        self.assertEqual(u.shape, self.shape)

        self.assertTrue((self.path / "adj_close.dat").exists())
        adj_close = self._open_memmap("adj_close", np.float32)
        self.assertEqual(adj_close.shape, self.shape)

        self.assertTrue((self.path / "adj_volume.dat").exists())
        adj_volume = self._open_memmap("adj_volume", np.float32)
        self.assertEqual(adj_volume.shape, self.shape)

        self.assertTrue((self.path / "mask.dat").exists())
        mask = self._open_memmap("mask", np.bool)
        self.assertEqual(mask.shape, self.shape)
        self.assertTrue(mask.all())


if __name__ == "__main__":
    unittest.main()
