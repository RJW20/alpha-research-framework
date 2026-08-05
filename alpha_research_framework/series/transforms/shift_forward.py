import numpy as np
import numpy.typing as npt
from numba import njit


@njit
def shift_forward(arr: npt.NDArray[np.floating], *, period: int) -> None:
    """
    Shift `arr` forward along axis 0 by `period` rows in-place.

    Vacated positions are filled with `np.nan`.
    """

    if period <= 0:
        return

    T = arr.shape[0]

    if period >= T:
        arr[:] = np.nan
        return

    for t in range(T - 1, period - 1, -1):
        arr[t] = arr[t - period]

    for t in range(period):
        arr[t] = np.nan
