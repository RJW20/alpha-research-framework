import unittest

import numpy as np

from alpha_research_framework.metrics.stats import pearson_scalar


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

    def test_constant(self) -> None:
        """Verify scalar pearson is nan for constant x."""

        x = np.ones(shape=self.SIZE)
        rng = np.random.default_rng(0)
        y = rng.uniform(0, self.SIZE, self.SIZE)
        np.testing.assert_equal(pearson_scalar(x, y), np.nan)



if __name__ == "__main__":
    unittest.main()
