import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import override

import numpy as np

import alpha_research_framework.market_data as md
from alpha_research_framework import (
    Alpha,
    EquityData,
    Universe,
    Window,
    evaluate,
)
from alpha_research_framework.universe import CrossSection
from tests.utils import (
    RegistryIsolatedTestCase,
    require_e2e_data_dir,
    set_e2e_data_dir,
)


class TestInformationLessAlphas(RegistryIsolatedTestCase):
    REGISTRY_OWNER = Alpha

    @classmethod
    def setUpClass(cls) -> None:
        """Require end-to-end directory and set up the universe."""

        data_path = require_e2e_data_dir()
        equity_data = EquityData(data_path)
        cls.tmp_dir = TemporaryDirectory()
        cls.universe = Universe(Path(cls.tmp_dir.name), equity_data, 0, 0)

    def test_random_noise(self) -> None:
        """Verify IC for random signal is ~0 to 2 decimal places."""

        class RandomNoise(Alpha):
            """
            Alpha that takes no inputs and gives a random signal per stock.
            """

            __rng__ = np.random.default_rng(0)

            ID = "random_noise"
            CATEGORY = "testing"
            DEPENDENCIES = set()
            HORIZONS = set(Window)

            @classmethod
            @override
            def compute(cls, x: CrossSection) -> md.Array:
                """`a_t ~ N[0,1]`"""
                return cls.__rng__.standard_normal(
                    size=x["price"].shape
                ).astype(md.Scalar)

        result = evaluate(self.universe, [RandomNoise])
        ic_df = result["information_coefficient"]
        ic_mean = ic_df.mean()
        np.testing.assert_array_almost_equal(
            ic_mean.to_numpy(),
            np.zeros(shape=ic_mean.shape),
            decimal=2,
        )

    def test_constant(self) -> None:
        """Verify IC for constant signal is nan."""

        class Constant(Alpha):
            """Alpha that gives a signal of 1 for every stock."""

            ID = "constant"
            CATEGORY = "testing"
            DEPENDENCIES = set()
            HORIZONS = set(Window)

            @classmethod
            @override
            def compute(cls, x: CrossSection) -> md.Array:
                """`a_t = 1`"""
                return np.ones_like(x["price"], dtype=md.Scalar)

        ic_df = evaluate(self.universe, [Constant])["information_coefficient"]
        ic_mean = ic_df.mean()
        np.testing.assert_array_almost_equal(
            ic_mean.to_numpy(),
            np.full_like(ic_mean, np.nan),
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.universe = None
        cls.tmp_dir.cleanup()


if __name__ == "__main__":
    set_e2e_data_dir()
    unittest.main(verbosity=2)
