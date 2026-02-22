import numpy as np


def extract_valid(*arrays: np.ndarray) -> None | tuple[np.ndarray, ...]:
    """
    Return all arrays (in the same order) masked to only include elements where
    none of them are np.nan, or None if there are none.
    Assumes all arrays have the same shape.
    """

    mask = ~np.isnan(arrays[0])
    for arr in arrays[1:]:
        mask &= ~np.isnan(arr)

    if not np.any(mask):
        return None

    return tuple(arr[mask] for arr in arrays)
