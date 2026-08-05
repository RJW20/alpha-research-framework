import unittest
from typing import Any, override

import alpha_research_framework.market_data as md
from alpha_research_framework.series.series import Series
from alpha_research_framework.series.transform import transform


class TestTransform(unittest.TestCase):

    def test_source_subtype(self) -> None:
        """Verify parent type of `source` asserted."""

        def identity(arr: md.Array) -> None:
            pass

        with self.assertRaises(TypeError):
            transform(str, func=identity)

    def test_target(self) -> None:
        """Verify a target cannot be transformed into a predictor."""

        class Predictor(Series):
            TAG = Series.Tag.PREDICTOR

        class Target(Series):
            TAG = Series.Tag.TARGET

        def identity(arr: md.Array) -> None:
            pass

        PredictorOnPredictor = transform(Predictor, target=False, func=identity)
        with self.assertRaises(ValueError):
            PredictorOnTarget = transform(Target, target=False, func=identity)
        TargetOnPredictor = transform(Predictor, target=True, func=identity)
        TargetOnTarget = transform(Target, target=True, func=identity)

    def test_return_source(self) -> None:
        """
        Verify returned `TransformedSeries.SOURCE` is set as the `Series` that
        is transformed.
        """

        class Base(Series):
            TAG = Series.Tag.PREDICTOR

        def identity(arr: md.Array) -> None:
            pass

        Transformed = transform(Base, func=identity)
        self.assertEqual(Transformed.SOURCE, Base)

    def test_return_transform(self) -> None:
        """
        Verify returned `TransformedSeries.TRANSFORM` is set as a partial
        function of `func` with any `kwargs` already applied.
        """

        class Dummy(Series):
            TAG = Series.Tag.PREDICTOR

        def raise_to_power(x: int, *, power: int) -> int:
            return x ** power

        Transformed = transform(Dummy, func=raise_to_power, power=2)
        self.assertEqual(Transformed.TRANSFORM(2), 4)


if __name__ == "__main__":
    unittest.main()
