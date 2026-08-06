import operator
import unittest

from alpha_research_framework.signals.signal import Signal


class TestSignalMetaCombine(unittest.TestCase):

    COMBINATIONS = [operator.add, operator.sub, operator.mul, operator.truediv]

    def test_non_subclass(self) -> None:
        """
        Verify cannot combine with a type that does not inherit from `Signal`.
        """

        class ValidSignal(Signal):
            pass

        class InvalidSignal:
            pass

        for combination in TestSignalMetaCombine.COMBINATIONS:
            with self.assertRaises(TypeError):
                combination(ValidSignal, InvalidSignal)

    def test_return_sources(self) -> None:
        """
        Verify returned `CombinedSignal`' `SOURCE_LEFT` and `SOURCE_RIGHT`
        are set as the `Signal`s that are combined.
        """

        class Left(Signal):
            pass

        class Right(Signal):
            pass

        for combination in TestSignalMetaCombine.COMBINATIONS:
            Combination = combination(Left, Right)
            self.assertEqual(Combination.SOURCE_LEFT, Left)
            self.assertEqual(Combination.SOURCE_RIGHT, Right)

    def test_return_binary_op(self) -> None:
        """
        Verify returned `CombinedsSignal.BINARY_OP` is set as correct
        `operator`.
        """

        class Left(Signal):
            pass

        class Right(Signal):
            pass

        for combination in TestSignalMetaCombine.COMBINATIONS:
            Combination = combination(Left, Right)
            self.assertIs(Combination.BINARY_OP, combination)


class TestSignalMetaNegate(unittest.TestCase):

    def test_return_sources(self) -> None:
        """
        Verify returned `NegatedSignal`'s `SOURCE` is set as the `Signal` that
        is negated.
        """

        class Source(Signal):
            pass

        Negated = - Source
        self.assertEqual(Negated.SOURCE, Source)


if __name__ == "__main__":
    unittest.main()
