import numpy as np
import numpy.typing as npt


def log(arr: npt.NDArray[np.floating]) -> None:
    """Apply `np.log` to `arr` in-place."""

    np.log(arr, out=arr)
