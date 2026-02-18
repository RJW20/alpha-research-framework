import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import numpy.typing as npt

from alpha_research_framework import (
    Alpha,
    EquityData,
    Universe,
    Window,
    evaluate,
)
from alpha_research_framework.features import FeatureSpec
from alpha_research_framework.universe import CrossSection
from tests.utils import require_e2e_data_dir, set_e2e_data_dir
    

class TestInformationLessAlphas(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        """Require end-to-end directory and set up the universe."""

        data_path = require_e2e_data_dir()
        equity_data = EquityData(data_path)
        cls.tmp_dir = TemporaryDirectory()
        cls.universe = Universe(
            Path(cls.tmp_dir.name),
            equity_data,
            0,
            0
        )

    def test_random_noise(self) -> None:
        """Verify IC for random signal is ~0 to 2 decimal places."""

        class RandomNoise(Alpha):
            """
            Alpha that takes no inputs and gives a random signal per stock.
            """

            __rng__ = np.random.default_rng(0)

            NAME = "random_noise"
            CATEGORY = "testing"
            HORIZONS = set(Window)

            def compute(self, x: CrossSection) -> npt.NDArray[np.float32]:
                """a_t ~ N[0,1]"""
                return self.__rng__.normal(
                    0,
                    1,
                    size=x["price"].shape
                ).astype(np.float32)
            
            def _init_dependencies(self) -> set[FeatureSpec]:
                return set()
            
        ic_df = evaluate(self.universe, [RandomNoise()])
        ic_mean = ic_df.mean()
        np.testing.assert_array_almost_equal(
            ic_mean.to_numpy(),
            np.zeros(shape=ic_mean.shape),
            decimal=2
        )

    def test_constant(self) -> None:
        """Verify IC for constant signal is nan."""

        class Constant(Alpha):
            """Alpha that gives a signal of 1 for every stock."""

            NAME = "constant"
            CATEGORY = "testing"
            HORIZONS = set(Window)

            def compute(self, x: CrossSection) -> npt.NDArray[np.float32]:
                """a_t = 1"""
                return np.ones_like(x["price"], dtype=np.float32)

            def _init_dependencies(self) -> set[FeatureSpec]:
                return set()
            
        ic_df = evaluate(self.universe, [Constant()])
        ic_mean = ic_df.mean()
        np.testing.assert_array_almost_equal(
            ic_mean.to_numpy(),
            np.full_like(ic_mean, np.nan)
        )
        
    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp_dir.cleanup()


if __name__ == "__main__":
    set_e2e_data_dir()
    unittest.main(verbosity=2)
