import os
import shutil
from pathlib import Path

from alpha_research_framework.scalar import Scalar

from .array import Array


class Allocator:
    """Allocate and release `Array`s with fixed location and shape."""

    def __init__(self, path: Path, shape: tuple[int, int]) -> None:
        """
        Remove any file or directory `path` refers to and then create a new
        directory.
        """

        shutil.rmtree(path, ignore_errors=True)
        path.mkdir(parents=True)

        self._path = path
        self._shape = shape

    def allocate(self, *, identifier: str) -> Array:
        """
        Return a new `Array` at `self._path/{identifier}.dat` with shape
        `self._shape`.
        """

        return Array(
            self._path / f"{identifier}.dat",
            dtype=Scalar,
            mode="w+",
            shape=self._shape,
        )

    def release(self, arr: Array) -> None:
        """Release `arr` if it exists."""

        if arr.filename:
            os.remove(arr.filename)
