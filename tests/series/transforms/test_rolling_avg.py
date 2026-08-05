import unittest

import numpy as np
import pandas as pd

from alpha_research_framework.series.transforms.rolling_avg import rolling_avg
from tests.utils import random_array


class TestRollingAvg(unittest.TestCase):

    SIZE = (1000, 100)

    def test_non_positive_lookback(self) -> None:
        """Verify a `ValueError` is thrown when `lookback <=0`."""

        arr = random_array(TestRollingAvg.SIZE)
        with self.assertRaises(ValueError):
            rolling_avg(arr, lookback=0)

    def test_compute(self) -> None:
        """
        Verify `arr` is modified to be the rolling average of itself.

        The result is compared to the `pandas` built-in `DataFrame` version.
        """

        for lookback in [1, 10, 100]:

            arr = random_array(TestRollingAvg.SIZE)
            expected = (
                pd.DataFrame(arr)
                .rolling(lookback, min_periods=1)
                .mean()
                .to_numpy(dtype=np.float64)
            )
            rolling_avg(arr, lookback=lookback)
            np.testing.assert_array_almost_equal(arr, expected)


if __name__ == "__main__":
    unittest.main()
