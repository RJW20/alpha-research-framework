import unittest

import numpy as np

import alpha_research_framework.observables as observables
import alpha_research_framework.series as series
from tests.utils import MockMdAllocator, random_array


class TestBuild(unittest.TestCase):

    def setUp(self) -> None:
        self.md = {
            observables.Open: random_array(MockMdAllocator.SIZE),
            observables.Close: random_array(MockMdAllocator.SIZE),
        }
        self.built = series.build(
            [series.Open, series.Close, series.LogClose],
            market_data=self.md,
            allocator=MockMdAllocator,
        )
        return super().setUp()

    def test_keys(self) -> None:
        """
        Verify all requested `Series` (and no more) are present in returned
        dict.
        """

        self.assertSetEqual(
            set(self.built.keys()),
            {series.Open, series.Close, series.LogClose},
        )

    def test_values(self) -> None:
        """Verify rudimental `Series` values' are correct in returned dict."""

        np.testing.assert_array_equal(
            self.built[series.Open],
            self.md[observables.Open],
        )
        np.testing.assert_array_equal(
            self.built[series.Close],
            self.md[observables.Close],
        )
        np.testing.assert_array_equal(
            self.built[series.LogClose],
            np.log(self.md[observables.Close]),
        )


if __name__ == "__main__":
    unittest.main()
