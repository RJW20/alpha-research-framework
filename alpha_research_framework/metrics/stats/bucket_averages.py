import numpy as np
import numpy.typing as npt
from numba import njit


@njit
def bucket_averages(
    bucket_idx: npt.NDArray[np.integer],
    x: np.ndarray,
    num_buckets: int
) -> np.ndarray:
    """
    Return the average per bucket in x specified by bucket_idx.
    
    Emtpy buckets will result in np.nan.
    """

    bucket_totals = np.zeros(num_buckets)
    bucket_counts = np.zeros(num_buckets)

    for i in range(len(bucket_idx)):
        b = bucket_idx[i]
        bucket_totals[b] += x[i]
        bucket_counts[b] += 1

    bucket_averages = np.empty(num_buckets)
    for b in range(num_buckets):
        if bucket_counts[b] > 0:
            bucket_averages[b] = bucket_totals[b] / bucket_counts[b]
        else:
            bucket_averages[b] = np.nan

    return bucket_averages
