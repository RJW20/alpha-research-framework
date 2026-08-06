import unittest
from typing import Any, override

import numpy as np

from alpha_research_framework.signals.negated_signal import NegatedSignal
from alpha_research_framework.signals.signal import Signal


class TestNegatedSignalSource(unittest.TestCase):

    def test_class_var_assertion(self) -> None:
        """
        Verify definition and type of `SOURCE` asserted when subclassing.
        """

        with self.assertRaises(AttributeError):
            class NoSource(NegatedSignal):
                pass
        
        with self.assertRaises(TypeError):
            class IncompatibleSource(NegatedSignal):
                SOURCE = str


class TestNegatedSignalCompute(unittest.TestCase):

    SIZE = 10

    def test_binary_op(self) -> None:
        """Verify `_compute` returns the negation of `SOURCE`."""

        class Ones(Signal):
            @classmethod
            @override
            def compute(cls, cross_section: Any, cache: Any) -> np.ndarray:
                return np.ones(TestNegatedSignalCompute.SIZE)

        class Negated(NegatedSignal):
            SOURCE = Ones

        result = Negated._compute({}, {})
        np.testing.assert_array_equal(
            result,
            [-1 for _ in range(TestNegatedSignalCompute.SIZE)],
        )


if __name__ == "__main__":
    unittest.main()
