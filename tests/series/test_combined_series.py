import operator
import unittest
from typing import Any, override

import numpy as np

from alpha_research_framework.series.combined_series import CombinedSeries
from alpha_research_framework.series.series import Series
from tests.utils import MockMdAllocator


class TestCombinedSeriesSourceLeft(unittest.TestCase):

    def test_class_var_assertion(self) -> None:
        """
        Verify definition and type of `SOURCE_LEFT` asserted when subclassing.
        """

        with self.assertRaises(AttributeError):
            class NoSourceLeft(CombinedSeries):
                TAG = Series.Tag.PREDICTOR
        
        with self.assertRaises(TypeError):
            class IncompatibleSourceLeft(CombinedSeries):
                TAG = Series.Tag.PREDICTOR
                SOURCE_LEFT = str


class TestCombinedSeriesSourceRight(unittest.TestCase):

    def test_class_var_assertion(self) -> None:
        """
        Verify definition and type of `SOURCE_RIGHT` asserted when subclassing.
        """

        class Dummy(Series):
            TAG = Series.Tag.PREDICTOR

        with self.assertRaises(AttributeError):
            class NoSourceRight(CombinedSeries):
                TAG = Series.Tag.PREDICTOR
                SOURCE_LEFT = Dummy
        
        with self.assertRaises(TypeError):
            class IncompatibleSourceRight(CombinedSeries):
                TAG = Series.Tag.PREDICTOR
                SOURCE_LEFT = Dummy
                SOURCE_RIGHT = str


class TestCombinedSeriesBinaryOp(unittest.TestCase):

    def test_class_var_assertion(self) -> None:
        """
        Verify definition, type and number of arguments of `BINARY_OP` asserted
        when subclassing.
        """

        class Dummy(Series):
            TAG = Series.Tag.PREDICTOR

        with self.assertRaises(AttributeError):
            class NoBinaryOp(CombinedSeries):
                TAG = Series.Tag.PREDICTOR
                SOURCE_LEFT = Dummy
                SOURCE_RIGHT = Dummy
        
        with self.assertRaises(TypeError):
            class IncompatibleBinaryOp(CombinedSeries):
                TAG = Series.Tag.PREDICTOR
                SOURCE_LEFT = Dummy
                SOURCE_RIGHT = Dummy
                BINARY_OP = 1

        with self.assertRaises(TypeError):
            class InvalidArgumentsBinaryOp(CombinedSeries):
                TAG = Series.Tag.PREDICTOR
                SOURCE_LEFT = Dummy
                SOURCE_RIGHT = Dummy
                BINARY_OP = lambda x: x * 2


class TestCombinedSeriesCompute(unittest.TestCase):

    def test_binary_op(self) -> None:
        """
        Verify `_compute` returns the correct binary combination of
        `SOURCE_LEFT` and `SOURCE_RIGHT`.
        """
        class Ones(Series):
            TAG = Series.Tag.PREDICTOR
            @classmethod
            @override
            def compute(
                cls,
                market_data: Any,
                cache: Any,
                allocator: MockMdAllocator,
            ) -> np.ndarray:
                return np.ones(allocator.SIZE)
            
        class Twos(Series):
            TAG = Series.Tag.PREDICTOR
            @classmethod
            @override
            def compute(
                cls,
                market_data: Any,
                cache: Any,
                allocator: MockMdAllocator,
            ) -> np.ndarray:
                return np.full(allocator.SIZE, 2)
            
        for binary_op in [
            operator.add, operator.sub, operator.mul, operator.truediv
        ]:

            class Combined(CombinedSeries):
                TAG = Series.Tag.PREDICTOR
                SOURCE_LEFT = Ones
                SOURCE_RIGHT = Twos
                BINARY_OP = binary_op

            result = Combined._compute({}, {}, MockMdAllocator)
            np.testing.assert_array_equal(
                result,
                [binary_op(1,2) for _ in range(MockMdAllocator.SIZE)],
            )


if __name__ == "__main__":
    unittest.main()
