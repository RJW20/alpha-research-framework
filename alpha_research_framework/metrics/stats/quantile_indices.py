import numpy as np
import numpy.typing as npt


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
    q_idx[order] = np.floor(np.arange(N) * num_quantiles / N).astype(int)

    return q_idx
