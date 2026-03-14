import unittest
from typing import override

import numpy as np

import alpha_research_framework.market_data as md
from alpha_research_framework.alphas.alpha.alpha import Alpha
from alpha_research_framework.universe import CrossSection


class TestCompose(unittest.TestCase):

    N = 500
    COMPOSITIONS = [
        lambda a, b: a + b,
        lambda a, b: a - b,
        lambda a, b: a * b,
        lambda a, b: a / b,
    ]

    def test_invalid(self) -> None:
        """
        Verify cannot compose with a type that does not inherit from `Alpha`.
        """

        class ValidAlpha(Alpha, abstract=True):
            pass

        class InvalidAlpha:
            pass

        for composition in TestCompose.COMPOSITIONS:
            with self.assertRaises(TypeError):
                class Composition(composition(ValidAlpha, InvalidAlpha)):
                    pass

    def test_dependencies(self) -> None:
        """Verify `DEPENDENCIES` is the union of bases' `DEPENDENCIES`."""

        class DependentOn1And2(Alpha, abstract=True):
            DEPENDENCIES = {1, 2}

        class DependentOn2And3(Alpha, abstract=True):
            DEPENDENCIES = {2, 3}

        for composition in TestCompose.COMPOSITIONS:
            
            class Composition(
                composition(DependentOn1And2, DependentOn2And3),
                abstract=True,
            ):
                pass

            self.assertSetEqual(Composition.DEPENDENCIES, {1, 2, 3})

    def test_compute(self) -> None:
        """Verify `compute` is the correct composition of bases' `compute`s."""

        class Ones(Alpha, abstract=True):
            DEPENDENCIES = set()
            @classmethod
            @override
            def compute(cls, x: CrossSection) -> md.Array:
                return np.ones(TestCompose.N, dtype=md.Scalar)
            
        class Twos(Alpha, abstract=True):
            DEPENDENCIES = set()
            @classmethod
            @override
            def compute(cls, x: CrossSection) -> md.Array:
                return np.full(TestCompose.N, 2, dtype=md.Scalar)
            
        for composition in TestCompose.COMPOSITIONS:
            
            class Composition(composition(Ones, Twos), abstract=True):
                pass

            np.testing.assert_array_equal(
                Composition.compute(None),
                composition(Ones.compute(None), Twos.compute(None)),
            )


if __name__ == "__main__":
    unittest.main()
