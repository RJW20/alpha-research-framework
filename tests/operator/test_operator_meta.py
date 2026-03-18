import unittest
from typing import TypeVar, override

from alpha_research_framework.operator.operator import Operator


class TestOperatorMeta(unittest.TestCase):

    def test_call_forwarding(self) -> None:
        """Verify `__call__` is overloaded to forward to `compute`."""

        class Add(Operator):

            T = TypeVar('T')
            @classmethod
            @override
            def compute(cls, x: T, y: T) -> T:
                return x + y
            
        self.assertEqual(Add(1,2), Add.compute(1, 2))


if __name__ == "__main__":
    unittest.main()
