import unittest

from alpha_research_framework.series.series import Series


class TestSeriesTag(unittest.TestCase):

    def test_class_var_assertion(self) -> None:
        """Verify definition and type of `TAG` asserted when subclassing."""

        with self.assertRaises(AttributeError):
            class NoTag(Series):
                pass

        with self.assertRaises(TypeError):
            class IncompatibleTag(Series):
                TAG = 1


if __name__ == "__main__":
    unittest.main()
