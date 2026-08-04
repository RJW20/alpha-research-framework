

class MockMdAllocator:

    SIZE = 5

    @staticmethod
    def allocate(identifier: str) -> list[int]:
        return [0] * MockMdAllocator.SIZE
