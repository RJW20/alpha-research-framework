from typing import TypeAlias

import numpy as np
import numpy.typing as npt

ShapeLike: TypeAlias = int | tuple[int, ...]

rng = np.random.default_rng(0)


def random_array(
    size: ShapeLike,
    *,
    inc_nans: bool = True,
) -> npt.NDArray[np.float64]:
    """
    Return a `NumPy` array of (seeded) randomly generated uniformly distributed
    values in [0, 1).

    The array will have given `size`.
    If `inc_nans` is set, approximately 10% of the values in the array will be
    `np.nan`.
    """

    arr = np.random.random(size)
    if inc_nans:
        nan_mask = rng.uniform(size=size) < 0.1
        arr[nan_mask] = np.nan
    return arr
