import numpy as np


class MockMdAllocator:

    SIZE = 5

    @staticmethod
    def allocate(identifier: str) -> np.ndarray:
        return np.zeros(MockMdAllocator.SIZE)
