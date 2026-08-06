import unittest

import numpy as np

import alpha_research_framework.series as series
from alpha_research_framework.universe.universe import Universe


class TestUniverse(unittest.TestCase):

    def test_cross_section_series(self) -> None:
        """
        Verify `_cross_section` returns all of the series with requested
        `tag`.
        """

        class Predictor1(series.Series):
            TAG = series.Series.Tag.PREDICTOR
            
        class Predictor2(series.Series):
            TAG = series.Series.Tag.PREDICTOR

        shape = (1, 1)
        mask = np.full(shape, True, dtype=bool)
        series_ = {
            Predictor1: np.ones_like(mask),
            Predictor2: np.ones_like(mask),
        }
        u = Universe(shape, None, mask, series_)

        predictor_xs = u._cross_section(0, series.Series.Tag.PREDICTOR)
        self.assertListEqual(
            list(predictor_xs.keys()),
            [Predictor1, Predictor2],
        )

        target_xs = u._cross_section(0, series.Series.Tag.TARGET)
        self.assertEqual(len(target_xs), 0)

    def test_cross_sections_mask(self) -> None:
        """Verify `_cross_section` masks series correctly."""

        class Predictor(series.Series):
            TAG = series.Series.Tag.PREDICTOR

        T = 10
        shape = (T, T)
        mask = np.array([[False] * t + [True] * (T - t) for t in range(T)])
        series_ = {Predictor: np.ones_like(mask)}
        u = Universe(shape, None, mask, series_)

        for t in range(T):
            predictor_xs = u._cross_section(t, series.Series.Tag.PREDICTOR)
            self.assertEqual(len(predictor_xs[Predictor]), T - t)


if __name__ == "__main__":
    unittest.main()
