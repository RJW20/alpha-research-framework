import numpy as np


class DataWrapper:
    """Thin container wrapping a dict[str, np.memmap]."""

    def __init__(self):
        self._data: dict[str, np.memmap] = {}

    @property
    def keys(self) -> set[str]:
        return set(key for key in self._data.keys())

    def add(self, name: str, values: np.memmap):
        self._data[name] = values

    def get(self, name: str) -> np.memmap:
        return self._data[name]
