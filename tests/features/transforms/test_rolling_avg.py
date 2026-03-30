import unittest

import numpy as np
import pandas as pd

from alpha_research_framework.features.transforms.rolling_avg import RollingAvg


class TestRollingAvg(unittest.TestCase):

    def test_compute(self) -> None:
        """
        Verify `arr` is modified to be the rolling average of itself.

        The result is compared to the `pandas` built-in `DataFrame` version.
        """

        SIZE = (1000, 100)
        rng = np.random.default_rng(0)

        for lookback in [1, 10, 100]:

            arr = rng.uniform(size=SIZE)
            nan_mask = rng.uniform(size=arr.shape) < 0.1
            arr[nan_mask] = np.nan

            expected = (
                pd.DataFrame(arr)
                .rolling(lookback, min_periods=1)
                .mean()
                .to_numpy(dtype=np.float64)
            )
            RollingAvg.compute(arr, lookback=lookback)

            np.testing.assert_array_almost_equal(arr, expected)


if __name__ == "__main__":
    unittest.main()
