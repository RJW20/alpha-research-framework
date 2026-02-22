import numpy as np
import numpy.typing as npt
from numba import njit


@njit
def quantile_indices(
    x: np.ndarray,
    num_quantiles: int
) -> npt.NDArray[np.integer]:
    """
    Return an array containing the quantile index for each element in x.
    
    Assumes x is clear of np.nan values.
    """

    N = len(x)
    order = np.argsort(x)
    q_idx = np.empty_like(order)
    for rank in range(N):
        q = (rank * num_quantiles) // N
        if q >= num_quantiles:
            q = num_quantiles - 1
        q_idx[order[rank]] = q

    return q_idx
