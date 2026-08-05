import unittest

import numpy as np
import pandas as pd

from alpha_research_framework.series.transforms.rolling_std import rolling_std
from tests.utils import random_array


class TestRollingStd(unittest.TestCase):

    SIZE = (1000, 100)

    def test_non_positive_lookback(self) -> None:
        """Verify a `ValueError` is thrown when `lookback <=0`."""

        arr = random_array(TestRollingStd.SIZE)
        with self.assertRaises(ValueError):
            rolling_std(arr, lookback=0)

    def test_values(self) -> None:
        """
        Verify `arr` is modified to be the rolling standard deviation of itself.

        The result is compared to the `pandas` built-in `DataFrame` version.
        """

        for lookback in [1, 10, 100]:

            arr = random_array(TestRollingStd.SIZE)
            expected = (
                pd.DataFrame(arr)
                .rolling(lookback, min_periods=1)
                .std()
                .to_numpy(dtype=np.float64)
            )
            rolling_std(arr, lookback=lookback)
            np.testing.assert_array_almost_equal(arr, expected)


if __name__ == "__main__":
    unittest.main()
