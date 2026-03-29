import unittest
from typing import Any, override

import alpha_research_framework.market_data as md
from alpha_research_framework.features.feature import Feature
from alpha_research_framework.features.transforms.transform import Transform


class TestTransformMetaCall(unittest.TestCase):

    def test_source_cls(self) -> None:
        """Verify parent type of `source_cls` asserted."""

        class Identity(Transform):
            @classmethod
            @override
            def compute(cls, arr: md.Array, **kwargs: Any) -> None:
                pass

        with self.assertRaises(TypeError):
            Identity(str)

    def test_target(self) -> None:
        """Verify a target cannot be transformed into a predictor."""

        class Predictor(Feature):
            TAG = Feature.Tag.PREDICTOR

        class Target(Feature):
            TAG = Feature.Tag.TARGET

        class Identity(Transform):
            @classmethod
            @override
            def compute(cls, arr: md.Array, **kwargs: Any) -> None:
                pass

        PredictorOnPredictor = Identity(Predictor, target=False)
        with self.assertRaises(ValueError):
            PredictorOnTarget = Identity(Target, target=False)
        TargetOnPredictor = Identity(Predictor, target=True)
        TargetOnTarget = Identity(Target, target=True)

    def test_source(self) -> None:
        """Verify `SOURCE` is set as the feature that is transformed."""

        class Base(Feature):
            TAG = Feature.Tag.PREDICTOR

        class Identity(Transform):
            @classmethod
            @override
            def compute(cls, arr: md.Array, **kwargs: Any) -> None:
                pass

        Derived = Identity(Base)
        self.assertEqual(Derived.SOURCE, Base)

    def test_compute(self) -> None:
        """
        Verify `compute` returns the transformation of the base's.
        
        Also verifys `kwargs` are passed to transformations `compute`.
        """

        class OneTwoThree(Feature):
            TAG = Feature.Tag.PREDICTOR
            @classmethod
            @override
            def compute(
                cls,
                market_data: Any,
                cache: Any,
                out: list[int],
            ) -> None:
                out[:] = [1, 2, 3]
            
        class RaiseToPower(Transform):
            @classmethod
            @override
            def compute(
                cls,
                x: list[int],
                *,
                power: int,
                **kwargs: Any,
            ) -> None:
                x[:] = [i**power for i in x]

        Derived = RaiseToPower(OneTwoThree, power=2)

        result = []
        Derived.compute({}, {}, result)
        self.assertEqual(result, [1, 4, 9])


if __name__ == "__main__":
    unittest.main()
