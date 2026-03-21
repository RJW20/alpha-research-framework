import unittest

from alpha_research_framework.observables import Observable


class TestObservableName(unittest.TestCase):

    def test_class_var_assertion(self) -> None:
        """Verify definition and type of `NAME` validated when subclassing."""

        with self.assertRaises(AttributeError):
            class NoName(Observable):
                pass

        with self.assertRaises(TypeError):
            class IncompatibleName(Observable):
                NAME = 1


if __name__ == "__main__":
    unittest.main()
