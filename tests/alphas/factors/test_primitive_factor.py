import unittest

import alpha_research_framework.features as features
from alpha_research_framework.alphas.factors.primitive_factor import (
    PrimitiveFactor,
)


class TestPrimitiveFactorFeature(unittest.TestCase):

    def test_class_var_assertion(self) -> None:
        """
        Verify definition and type of `FEATURE` asserted when subclassing.
        """

        with self.assertRaises(AttributeError):
            class NoFeature(PrimitiveFactor):
                pass

        with self.assertRaises(TypeError):
            class IncompatibleFeature(PrimitiveFactor):
                FEATURE = PrimitiveFactor


class TestPrimitiveFactorCompute(unittest.TestCase):

    def test_insufficient_cross_section(self) -> None:
        """
        Verify a `ValueError` is thrown if the cross-section doesn't contain
        `FEATURE`.
        """

        class Required(features.Feature, abstract=True):
            pass

        class Requires(PrimitiveFactor):
            FEATURE = Required

        with self.assertRaises(ValueError):
            Requires.compute({}, {})

    def test_extracted_values(self) -> None:
        """Verify values extracted from cross-section pertain to `FEATURE`."""

        class Required(features.Feature, abstract=True):
            pass

        class Requires(PrimitiveFactor):
            FEATURE = Required

        required_values = ["a", "b"]
        x = {Required: required_values}

        self.assertListEqual(Requires.compute(x, {}), required_values)


if __name__ == "__main__":
    unittest.main()
