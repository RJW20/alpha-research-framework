import unittest

import numpy as np

from alpha_research_framework.features.transforms.lagged_difference import (
    LaggedDifference,
)


class TestLaggedDifference(unittest.TestCase):

    def test_compute(self) -> None:
        """Verify `arr` is modified to be the lagged difference of itself."""

        SIZE = 1000
        rng = np.random.default_rng(0)

        for lag in [1, 10, 100]:

            arr = rng.uniform(size=SIZE)
            nan_mask = rng.uniform(size=arr.shape) < 0.1
            arr[nan_mask] = np.nan

            expected = np.concatenate(
                [
                    np.full(lag, np.nan),
                    arr[lag:] - arr[:-lag],
                ]
            )
            LaggedDifference.compute(arr, lag=lag)
            
            np.testing.assert_array_equal(arr, expected)
        

if __name__ == "__main__":
    unittest.main()
