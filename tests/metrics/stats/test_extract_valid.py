import unittest

import numpy as np

from alpha_research_framework.metrics.stats import extract_valid


class TestExtractValid(unittest.TestCase):

    def test_single(self) -> None:
        """
        Verify when a single array is passed the mask simply removes np.nan
        values.
        """

        rng = np.random.default_rng(0)
        x = rng.uniform(size=100)
        nan_mask = rng.uniform(size=x.shape) < 0.1
        x[nan_mask] = np.nan
        np.testing.assert_array_equal(extract_valid(x)[0], x[~nan_mask])

    def test_multiple(self) -> None:
        """
        Verify when multiple arrays are passed the mask removes elements where
        any of them are np.nan.
        """

        N = 100
        rng = np.random.default_rng(0)
        arrays: list[np.ndarray] = []
        nan_mask_intersection = np.full(100, False)
        for i in range(N):
            arr = rng.uniform(size=N)
            nan_mask = rng.uniform(size=arr.shape) < 0.01
            arr[nan_mask] = np.nan
            arrays.append(arr)
            nan_mask_intersection |= nan_mask
        for valid, arr in zip(extract_valid(*arrays), arrays):
            np.testing.assert_array_equal(valid, arr[~nan_mask_intersection])

    def test_none(self) -> None:
        """
        Verify None is returned when the intersection of non np.nan values is
        empty.
        """

        N = 100
        rng = np.random.default_rng(0)
        arrays: list[np.ndarray] = []
        for i in range(N):
            arr = rng.uniform(size=N)
            arr[i] = np.nan
            arrays.append(arr)
        self.assertIsNone(extract_valid(*arrays))


if __name__ == "__main__":
    unittest.main()
