import unittest

import numpy as np

from alpha_research_framework.metrics.stats import quantile_indices


class TestQuantileIndices(unittest.TestCase):

    SIZE = 100

    def test_indices(self) -> None:
        """
        Verify quantile indices are same as sorted indices / number of
        quantiles.
        """

        Q = 10
        rng = np.random.default_rng(0)
        x = np.arange(self.SIZE)
        rng.shuffle(x)
        q_idx = quantile_indices(x, int(Q))
        sorted_indices = np.argsort(np.argsort(x))
        np.testing.assert_array_equal(
            q_idx,
            np.floor(sorted_indices / Q).astype(int)
        )

    def test_one_quantile(self) -> None:
        """Verify one quantile places all elements in quantile 0."""

        rng = np.random.default_rng(0)
        x = rng.integers(0, self.SIZE, self.SIZE)
        q_idx = quantile_indices(x, 1)
        np.testing.assert_array_equal(q_idx, np.zeros_like(x))

    def test_ties(self) -> None:
        """Verify ties don't alter the number of elements per quantile."""

        Q = 8
        x_ties = np.ones(self.SIZE)
        x = np.arange(self.SIZE)
        np.testing.assert_array_equal(
            np.bincount(quantile_indices(x_ties, Q)),
            np.bincount(quantile_indices(x, Q))
        )


if __name__ == "__main__":
    unittest.main()
