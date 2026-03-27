import unittest

import alpha_research_framework.features as features
from alpha_research_framework.universe.universe import Universe

import numpy as np


class TestUniverse(unittest.TestCase):

    def test_cross_section_features(self) -> None:
        """
        Verify `_cross_section` returns all of the features with requested
        `tag`.
        """

        class Predictor1(features.Feature):
            TAG = features.Feature.Tag.PREDICTOR
            
        class Predictor2(features.Feature):
            TAG = features.Feature.Tag.PREDICTOR

        shape = (1, 1)
        mask = np.full(shape, True, dtype=bool)
        features_ = {
            Predictor1: np.ones_like(mask),
            Predictor2: np.ones_like(mask),
        }
        u = Universe(shape, None, mask, features_)

        predictor_xs = u._cross_section(0, features.Feature.Tag.PREDICTOR)
        self.assertListEqual(
            list(predictor_xs.keys()),
            [Predictor1, Predictor2],
        )

        target_xs = u._cross_section(0, features.Feature.Tag.TARGET)
        self.assertEqual(len(target_xs), 0)

    def test_cross_sections_mask(self) -> None:
        """Verify `_cross_section` masks features correctly."""

        class Predictor(features.Feature):
            TAG = features.Feature.Tag.PREDICTOR

        T = 10
        shape = (T, T)
        mask = np.array([[False] * t + [True] * (T - t) for t in range(T)])
        features_ = {Predictor: np.ones_like(mask)}
        u = Universe(shape, None, mask, features_)

        for t in range(T):
            predictor_xs = u._cross_section(t, features.Feature.Tag.PREDICTOR)
            self.assertEqual(len(predictor_xs[Predictor]), T - t)


if __name__ == "__main__":
    unittest.main()
