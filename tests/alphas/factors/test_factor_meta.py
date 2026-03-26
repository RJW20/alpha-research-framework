import operator
import unittest
from typing import Any, override

import alpha_research_framework.features as features
from alpha_research_framework.alphas.factors.factor import Factor


class TestFactorMetaCombine(unittest.TestCase):

    COMBINATIONS = [operator.add, operator.sub, operator.mul, operator.truediv]

    def test_non_subclass(self) -> None:
        """
        Verify cannot combine with a type that does not inherit from `Factor`.
        """

        class ValidFactor(Factor, abstract=True):
            pass

        class InvalidFactor:
            pass

        for combination in TestFactorMetaCombine.COMBINATIONS:
            with self.assertRaises(TypeError):
                combination(ValidFactor, InvalidFactor)

    def test_required_features(self) -> None:
        """Verify `REQUIRED_FEATURES` is the union of the bases'."""

        class Feature1(features.Feature, abstract=True):
            pass

        class Feature2(features.Feature, abstract=True):
            pass

        class Feature3(features.Feature, abstract=True):
            pass

        class Requires1And2(Factor):
            REQUIRED_FEATURES = {Feature1, Feature2}

        class Requires2And3(Factor):
            REQUIRED_FEATURES = {Feature2, Feature3}

        for combination in TestFactorMetaCombine.COMBINATIONS:
            combined_factor = combination(Requires1And2, Requires2And3)
            self.assertSetEqual(
                combined_factor.REQUIRED_FEATURES,
                {Feature1, Feature2, Feature3},
            )

    def test_compute(self) -> None:
        """Verify `compute` is the correct combination of the bases'."""

        class One(Factor):
            REQUIRED_FEATURES = set()
            @classmethod
            @override
            def compute(cls, *args: Any, **kwargs: Any) -> Any:
                return 1
            
        class Two(Factor):
            REQUIRED_FEATURES = set()
            @classmethod
            @override
            def compute(cls, *args: Any, **kwargs: Any) -> Any:
                return 2
            
        for combination in TestFactorMetaCombine.COMBINATIONS:
            combined_factor = combination(One, Two)
            self.assertEqual(
                combined_factor.compute(None, {}),
                combination(1, 2),
            )


class TestFactorMetaNegate(unittest.TestCase):

    def test_required_features(self) -> None:
        """Verify `REQUIRED_FEATURES` is identical to the original's."""

        class Required(features.Feature, abstract=True):
            pass

        class Requires(Factor):
            REQUIRED_FEATURES = {Required}

        negated_factor = - Requires
        self.assertSetEqual(negated_factor.REQUIRED_FEATURES, {Required})

    def test_compute(self) -> None:
        """Verify `compute` is the negation of the original`s."""

        class One(Factor):
            REQUIRED_FEATURES = set()
            @classmethod
            @override
            def compute(cls, *args: Any, **kwargs: Any) -> Any:
                return 1
            
        negated_factor = - One
        self.assertEqual(negated_factor.compute({}, {}), -1)


if __name__ == "__main__":
    unittest.main()
