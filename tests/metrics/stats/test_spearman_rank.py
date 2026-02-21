import unittest

import numpy as np
import pandas as pd

from alpha_research_framework.metrics.stats import spearman_rank


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

    def test_nan(self) -> None:
        """Verify spearman rank is nan for full nan mask."""

        rng = np.random.default_rng(0)
        x = rng.uniform(0, self.SIZE, self.SIZE)
        y = rng.uniform(0, self.SIZE, self.SIZE)
        nan_mask = rng.uniform(size=self.SIZE) < 0.5
        x[nan_mask], y[~nan_mask] = np.nan, np.nan
        np.testing.assert_equal(spearman_rank(x, y), np.nan)

    def test_sign_flip(self) -> None:
        """Verify reversing ranking flips spearman rank."""

        x = np.array([1,2,3,4,5])
        y = np.array([5,4,3,2,1])
        self.assertEqual(spearman_rank(x,y), spearman_rank(-x,y) * -1)


if __name__ == "__main__":
    unittest.main()
