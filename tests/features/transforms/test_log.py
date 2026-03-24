import unittest

import numpy as np

from alpha_research_framework.features.transforms.log import Log


class TestLog(unittest.TestCase):

    def test_compute(self) -> None:
        """Verify `arr` is modified to be the log of itself."""

        SIZE = 1000
        rng = np.random.default_rng(0)

        arr = rng.uniform(size=SIZE)
        nan_mask = rng.uniform(size=arr.shape) < 0.1
        arr[nan_mask] = np.nan

        expected = np.log(arr)
        Log.compute(arr)
        
        np.testing.assert_array_equal(arr, expected)
        

if __name__ == "__main__":
    unittest.main()
