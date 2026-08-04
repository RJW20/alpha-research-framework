import unittest
from typing import Any, override

from alpha_research_framework.series.series import Series
from alpha_research_framework.series.transformed_series import TransformedSeries
from tests.utils import MockMdAllocator


class TestTransformedSeriesSource(unittest.TestCase):

    def test_class_var_assertion(self) -> None:
        """Verify definition and type of `SOURCE` asserted when subclassing."""

        with self.assertRaises(AttributeError):
            class NoSource(TransformedSeries):
                TAG = Series.Tag.PREDICTOR
        
        with self.assertRaises(TypeError):
            class IncompatibleSource(TransformedSeries):
                TAG = Series.Tag.PREDICTOR
                SOURCE = str


class TestTransformedSeriesTransform(unittest.TestCase):

    def test_class_var_assertion(self) -> None:
        """
        Verify definition, type and number of arguments of `TRANSFORM` asserted
        when subclassing.
        """

        class Dummy(Series):
            TAG = Series.Tag.PREDICTOR

        with self.assertRaises(AttributeError):
            class NoTransform(TransformedSeries):
                TAG = Series.Tag.PREDICTOR
                SOURCE = Dummy
        
        with self.assertRaises(TypeError):
            class IncompatibleTransform(TransformedSeries):
                TAG = Series.Tag.PREDICTOR
                SOURCE = Dummy
                TRANSFORM = 1

        with self.assertRaises(TypeError):
            class InvalidArgumentsTransform(TransformedSeries):
                TAG = Series.Tag.PREDICTOR
                SOURCE = Dummy
                TRANSFORM = lambda x, y: x * y


class TestTransformedSeriesCompute(unittest.TestCase):

    def test_transform(self) -> None:
        """Verify `_compute` returns the transformation of `SOURCE`'s."""

        class Integers(Series):
            TAG = Series.Tag.PREDICTOR
            @classmethod
            @override
            def compute(
                cls,
                market_data: Any,
                cache: Any,
                allocator: MockMdAllocator,
            ) -> list[int]:
                arr = allocator.allocate(None)
                arr = [i for i, a in enumerate(arr)]
                return arr

        def square(arr: list[int]) -> None:
            arr[:] = [x**2 for x in arr]

        class Transformed(TransformedSeries):
            TAG = Series.Tag.PREDICTOR
            SOURCE = Integers
            TRANSFORM = square

        result = Transformed._compute({}, {}, MockMdAllocator)
        self.assertEqual(result, [x**2 for x in range(MockMdAllocator.SIZE)])


if __name__ == "__main__":
    unittest.main()
