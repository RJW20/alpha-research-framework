import unittest

from alpha_research_framework.features.feature import Feature


class TestFeatureTag(unittest.TestCase):

    def test_class_var_assertion(self) -> None:
        """Verify definition and type of `TAG` asserted when subclassing."""

        with self.assertRaises(AttributeError):
            class NoTag(Feature):
                pass

        with self.assertRaises(TypeError):
            class IncompatibleTag(Feature):
                TAG = 1


if __name__ == "__main__":
    unittest.main()
