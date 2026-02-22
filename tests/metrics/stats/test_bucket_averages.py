import unittest

import numpy as np

from alpha_research_framework.metrics.stats import bucket_averages


class TestBucketAverages(unittest.TestCase):

    def test_basic_buckets(self) -> None:
        """Verify output for basic case."""

        bucket_idx = np.array([0, 1, 0, 1, 2])
        x = np.array([1, 2, 3, 4, 5])
        num_buckets = 3
        np.testing.assert_array_equal(
            bucket_averages(bucket_idx, x, num_buckets),
            np.array([(1 + 3) / 2, (2 + 4) / 2, 5])
        )

    def test_empty_bucket(self) -> None:
        """Verify empty buckets average to np.nan."""

        bucket_idx = np.array([0, 2])
        x = np.array([1, 3])
        num_buckets = 4
        np.testing.assert_array_equal(
            bucket_averages(bucket_idx, x, num_buckets),
            np.array([1, np.nan, 3, np.nan])
        )


if __name__ == "__main__":
    unittest.main()
