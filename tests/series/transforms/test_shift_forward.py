import unittest

import numpy as np

from alpha_research_framework.series.transforms.shift_forward import (
    shift_forward,
)
from tests.utils import random_array


class TestShiftForward(unittest.TestCase):

    SIZE = 10

    def test_non_positive_period(self) -> None:
        """Verify `arr` is unchanged when `period <= 0`."""

        arr = random_array(TestShiftForward.SIZE, inc_nans=False)
        expected = arr.copy()
        shift_forward(arr, period=0)
        np.testing.assert_array_equal(arr, expected)

    def test_period_too_large(self) -> None:
        """Verify `arr` is entirely `np.nan` when `period >= len(arr)`."""

        arr = random_array(TestShiftForward.SIZE, inc_nans=False)
        expected = np.full(TestShiftForward.SIZE, fill_value=np.nan)
        shift_forward(arr, period=TestShiftForward.SIZE + 1)
        np.testing.assert_array_equal(arr, expected)

    def test_unit_period(self) -> None:
        """Verify `arr` is shifted forward once when `period = 1`."""

        arr = random_array(TestShiftForward.SIZE, inc_nans=False)
        expected = np.concatenate(([np.nan], arr[:-1]))
        shift_forward(arr, period=1)
        np.testing.assert_array_equal(arr, expected)

    def test_standard_period(self) -> None:
        """
        Verfiy `arr` is shifted forward period steps for
        `1 < period < len(arr)`.
        """

        for period in [2, self.SIZE // 2]:
            arr = random_array(TestShiftForward.SIZE, inc_nans=False)
            expected = np.concatenate(([np.nan] * period, arr[:-period]))
            shift_forward(arr, period=period)
            np.testing.assert_array_equal(arr, expected)

if __name__ == "__main__":
    unittest.main()
