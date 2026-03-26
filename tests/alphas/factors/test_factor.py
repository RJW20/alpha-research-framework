import unittest

from alpha_research_framework.alphas.factors.factor import Factor


class TestFactorRequiredFeatures(unittest.TestCase):

    def test_class_var_assertion(self) -> None:
        """
        Verify definition and type of `REQUIRED_FEATURES` asserted when
        subclassing.
        """

        with self.assertRaises(AttributeError):
            class NoRequiredFeatures(Factor):
                pass

        with self.assertRaises(TypeError):
            class IncompatibleRequiredFeaturesContainer(Factor):
                REQUIRED_FEATURES = list()

        with self.assertRaises(TypeError):
            class IncompatibleRequiredFeaturesElement(Factor):
                REQUIRED_FEATURES = {Factor}


if __name__ == "__main__":
    unittest.main()
