import unittest

import numpy as np

from alpha_research_framework.series.transforms.shift_back import shift_back
from tests.utils import random_array


class TestShiftBack(unittest.TestCase):

    SIZE = 10

    def test_non_positive_period(self) -> None:
        """Verify `arr` is unchanged when `period <= 0`."""

        arr = random_array(TestShiftBack.SIZE, inc_nans=False)
        expected = arr.copy()
        shift_back(arr, period=0)
        np.testing.assert_array_equal(arr, expected)

    def test_period_too_large(self) -> None:
        """Verify `arr` is entirely `np.nan` when `period >= len(arr)`."""

        arr = random_array(TestShiftBack.SIZE, inc_nans=False)
        expected = np.full(TestShiftBack.SIZE, fill_value=np.nan)
        shift_back(arr, period=TestShiftBack.SIZE + 1)
        np.testing.assert_array_equal(arr, expected)

    def test_unit_period(self) -> None:
        """Verify `arr` is shifted back once when `period = 1`."""

        arr = random_array(TestShiftBack.SIZE, inc_nans=False)
        expected = np.concatenate((arr[1:], [np.nan]))
        shift_back(arr, period=1)
        np.testing.assert_array_equal(arr, expected)

    def test_standard_period(self) -> None:
        """
        Verfiy `arr` is shifted back period steps for `1 < period < len(arr)`.
        """

        for period in [2, self.SIZE // 2]:
            arr = random_array(TestShiftBack.SIZE, inc_nans=False)
            expected = np.concatenate((arr[period:], [np.nan] * period))
            shift_back(arr, period=period)
            np.testing.assert_array_equal(arr, expected)


if __name__ == "__main__":
    unittest.main()
