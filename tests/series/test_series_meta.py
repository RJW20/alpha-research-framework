import operator
import unittest

from alpha_research_framework.series.series import Series


class TestSeriesMetaCombine(unittest.TestCase):

    COMBINATIONS = [operator.add, operator.sub, operator.mul, operator.truediv]

    def test_non_subclass(self) -> None:
        """
        Verify cannot combine with a type that does not inherit from `Series`.
        """

        class ValidSeries(Series, abstract=True):
            pass

        class InvalidSeries:
            pass

        for combination in TestSeriesMetaCombine.COMBINATIONS:
            with self.assertRaises(TypeError):
                combination(ValidSeries, InvalidSeries)

    def test_return_tag(self) -> None:
        """
        Verify returned `TransformedSeries.TAG` is the logical maximum of the
        input `Series`' `TAG`s.
        """

        class Predictor(Series):
            TAG = Series.Tag.PREDICTOR

        class Target(Series):
            TAG = Series.Tag.TARGET

        for combination in TestSeriesMetaCombine.COMBINATIONS:
            self.assertEqual(
                combination(Predictor, Predictor).TAG,
                Series.Tag.PREDICTOR,
            )
            self.assertEqual(
                combination(Predictor, Target).TAG,
                Series.Tag.TARGET,
            )
            self.assertEqual(
                combination(Target, Predictor).TAG,
                Series.Tag.TARGET,
            )
            self.assertEqual(
                combination(Target, Target).TAG,
                Series.Tag.TARGET,
            )

    def test_return_sources(self) -> None:
        """
        Verify returned `CombinedSeries`' `SOURCE_LEFT` and `SOURCE_RIGHT`
        are set as the `Series` that are combined.
        """

        class Left(Series):
            TAG = Series.Tag.PREDICTOR

        class Right(Series):
            TAG = Series.Tag.PREDICTOR

        for combination in TestSeriesMetaCombine.COMBINATIONS:
            Combination = combination(Left, Right)
            self.assertEqual(Combination.SOURCE_LEFT, Left)
            self.assertEqual(Combination.SOURCE_RIGHT, Right)

    def test_return_binary_op(self) -> None:
        """
        Verify returned `CombinedSeries.BINARY_OP` is set as correct
        `operator`.
        """

        class Left(Series):
            TAG = Series.Tag.PREDICTOR

        class Right(Series):
            TAG = Series.Tag.PREDICTOR

        for combination in TestSeriesMetaCombine.COMBINATIONS:
            Combination = combination(Left, Right)
            self.assertIs(Combination.BINARY_OP, combination)


if __name__ == "__main__":
    unittest.main()
