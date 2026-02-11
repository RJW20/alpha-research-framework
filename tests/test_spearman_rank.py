import unittest

import numpy as np
import pandas as pd

from alpha_research_framework.spearman_rank import (
    pearson_scalar,
    rankdata,
    spearman_rank,
)


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


class TestPearsonScalar(unittest.TestCase):

    SIZE = 100

    def test(self) -> None:
        """
        Verify scalar pearson calculation is same as numpy built in covariant
        matrix version.
        """

        rng = np.random.default_rng(0)
        x = rng.uniform(0, self.SIZE, self.SIZE)
        y = rng.uniform(0, self.SIZE, self.SIZE)
        pearson = pearson_scalar(x, y)
        expected = np.corrcoef(x, y)[0, 1]
        self.assertAlmostEqual(pearson, expected)


class TestSpearmanRank(unittest.TestCase):
    
    SIZE = 100

    def test(self) -> None:
        """
        Verify spearman rank calculation is same as pandas built in dataframe
        version.
        """

        rng = np.random.default_rng(0)
        x = rng.uniform(0, self.SIZE, self.SIZE)
        y = rng.uniform(0, self.SIZE, self.SIZE)
        nan_mask_x = rng.uniform(size=self.SIZE) < 0.1
        nan_mask_y = rng.uniform(size=self.SIZE) < 0.1
        x[nan_mask_x], y[nan_mask_y] = np.nan, np.nan
        spearman = spearman_rank(x, y)
        df = pd.DataFrame({'x': x, 'y':y})
        expected = df.corr('spearman')['x']['y']
        self.assertAlmostEqual(spearman, expected)


if __name__ == "__main__":
    unittest.main()
