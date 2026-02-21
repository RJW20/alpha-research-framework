import unittest

import numpy as np

from alpha_research_framework.metrics.stats import rankdata


class TestRankData(unittest.TestCase):

    SIZE = 100

    def test_in_order(self) -> None:
        """Verify stricty increasing data is ranked by index."""

        x = np.arange(self.SIZE)
        x_ranks = rankdata(x)
        np.testing.assert_array_equal(x_ranks, x)

    def test_unique_data(self) -> None:
        """Verify unique data ranking is the same as sorted index."""

        rng = np.random.default_rng(0)
        x = np.arange(self.SIZE) * rng.integers(0, 10)
        rng.shuffle(x)
        x_ranks = rankdata(x)
        sorted_indices = np.argsort(np.argsort(x))
        np.testing.assert_array_equal(x_ranks, sorted_indices)

    def test_averaging(self) -> None:
        """Verify ranking averaging for non-unique data."""

        x = np.array([5, 2, 2, 4, 5])
        x_ranks = rankdata(x)
        expected = np.array([3.5, 0.5, 0.5, 2, 3.5])
        np.testing.assert_array_equal(x_ranks, expected)

        x = np.ones(self.SIZE)
        x_ranks = rankdata(x)
        expected = np.ones(self.SIZE) * (self.SIZE - 1) / 2
        np.testing.assert_array_equal(x_ranks, expected)


if __name__ == "__main__":
    unittest.main()
