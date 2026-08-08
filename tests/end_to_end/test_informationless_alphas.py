import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import override

import numpy as np

import alpha_research_framework as arf
import alpha_research_framework.cross_section as xs
from alpha_research_framework.scalar import Scalar
from tests.utils import (
    RegistryIsolatedTestCase,
    require_e2e_data_dir,
    set_e2e_data_dir,
)


class TestInformationLessAlphas(RegistryIsolatedTestCase):
    REGISTRY_OWNER = arf.alphas.Alpha

    def setUp(self) -> None:
        """Require end-to-end directory and create a temporary directory."""

        self.data_src = require_e2e_data_dir()
        self._tmp_dir = TemporaryDirectory()
        self.universe_path = Path(self._tmp_dir.name)

    def tearDown(self) -> None:
        self._tmp_dir.cleanup()

    def test_random_noise(self) -> None:
        """Verify IC for random signal is ~0 to 2 decimal places."""

        class RandomNoiseSignal(arf.signals.SeriesSignal):
            __rng__ = np.random.default_rng(0)
            SERIES = arf.series.Close
            @classmethod
            @override
            def compute(
                cls,
                cross_section: xs.CrossSection,
                cache: arf.signals.Cache,
            ) -> xs.Array:
                return cls.__rng__.standard_normal(
                    size=cross_section[arf.series.Close].shape,
                    dtype=Scalar,
                )

        class RandomNoise(arf.alphas.Alpha):
            ID = "random_noise"
            CATEGORY = "testing"
            SIGNAL = RandomNoiseSignal
            HORIZONS = set(arf.Window)

        universe = arf.build_universe_for(
            [RandomNoise],
            src=self.data_src,
            path=self.universe_path,
            liquidity_threshold=0,
            mcap_threshold=0,
        )

        result = arf.evaluate(universe, [RandomNoise])
        df = result[RandomNoise.ID]
        df_mean = df.mean()
        np.testing.assert_array_almost_equal(
            df_mean.to_numpy(),
            np.zeros(shape=df_mean.shape),
            decimal=2,
        )

    def test_constant(self) -> None:
        """Verify IC for constant signal is nan."""

        class ConstantSignal(arf.signals.SeriesSignal):
            SERIES = arf.series.Close
            @classmethod
            @override
            def compute(
                cls,
                cross_section: xs.CrossSection,
                cache: arf.signals.Cache,
            ) -> xs.Array:
                return np.ones_like(
                    cross_section[arf.series.Close],
                    dtype=Scalar,
                )

        class Constant(arf.alphas.Alpha):
            ID = "constant"
            CATEGORY = "testing"
            SIGNAL = ConstantSignal
            HORIZONS = set(arf.Window)

        universe = arf.build_universe_for(
            [Constant],
            src=self.data_src,
            path=self.universe_path,
            liquidity_threshold=0,
            mcap_threshold=0,
        )

        result = arf.evaluate(universe, [Constant])
        df = result[Constant.ID]
        df_mean = df.mean()
        np.testing.assert_array_equal(
            df_mean.to_numpy(),
            np.full_like(df_mean, np.nan),
        )


if __name__ == "__main__":
    set_e2e_data_dir()
    unittest.main(verbosity=2)
