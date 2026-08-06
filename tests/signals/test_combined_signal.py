import operator
import unittest
from typing import Any, override

import numpy as np

from alpha_research_framework.signals.combined_signal import CombinedSignal
from alpha_research_framework.signals.signal import Signal


class TestCombinedSignalSourceLeft(unittest.TestCase):

    def test_class_var_assertion(self) -> None:
        """
        Verify definition and type of `SOURCE_LEFT` asserted when subclassing.
        """

        with self.assertRaises(AttributeError):
            class NoSourceLeft(CombinedSignal):
                pass
        
        with self.assertRaises(TypeError):
            class IncompatibleSourceLeft(CombinedSignal):
                SOURCE_LEFT = str


class TestCombinedSignalSourceRight(unittest.TestCase):

    def test_class_var_assertion(self) -> None:
        """
        Verify definition and type of `SOURCE_RIGHT` asserted when subclassing.
        """

        class Dummy(Signal):
            pass

        with self.assertRaises(AttributeError):
            class NoSourceRight(CombinedSignal):
                SOURCE_LEFT = Dummy
        
        with self.assertRaises(TypeError):
            class IncompatibleSourceRight(CombinedSignal):
                SOURCE_LEFT = Dummy
                SOURCE_RIGHT = str


class TestCombinedSignalBinaryOp(unittest.TestCase):

    def test_class_var_assertion(self) -> None:
        """
        Verify definition, type and number of arguments of `BINARY_OP` asserted
        when subclassing.
        """

        class Dummy(Signal):
            pass

        with self.assertRaises(AttributeError):
            class NoBinaryOp(CombinedSignal):
                pass
                SOURCE_LEFT = Dummy
                SOURCE_RIGHT = Dummy
        
        with self.assertRaises(TypeError):
            class IncompatibleBinaryOp(CombinedSignal):
                pass
                SOURCE_LEFT = Dummy
                SOURCE_RIGHT = Dummy
                BINARY_OP = 1

        with self.assertRaises(TypeError):
            class InvalidArgumentsBinaryOp(CombinedSignal):
                pass
                SOURCE_LEFT = Dummy
                SOURCE_RIGHT = Dummy
                BINARY_OP = lambda x: x * 2


class TestCombinedSignalCompute(unittest.TestCase):

    SIZE = 10

    def test_binary_op(self) -> None:
        """
        Verify `_compute` returns the correct binary combination of
        `SOURCE_LEFT` and `SOURCE_RIGHT`.
        """
        
        class Ones(Signal):
            @classmethod
            @override
            def compute(cls, cross_section: Any, cache: Any) -> np.ndarray:
                return np.ones(TestCombinedSignalCompute.SIZE)
            
        class Twos(Signal):
            @classmethod
            @override
            def compute(cls, cross_section: Any, cache: Any) -> np.ndarray:
                return np.full(TestCombinedSignalCompute.SIZE, 2)
            
        for binary_op in [
            operator.add, operator.sub, operator.mul, operator.truediv
        ]:

            class Combined(CombinedSignal):
                SOURCE_LEFT = Ones
                SOURCE_RIGHT = Twos
                BINARY_OP = binary_op

            result = Combined._compute({}, {})
            np.testing.assert_array_equal(
                result,
                [binary_op(1,2) for _ in range(TestCombinedSignalCompute.SIZE)],
            )


if __name__ == "__main__":
    unittest.main()
