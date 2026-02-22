import numpy as np
import numpy.typing as npt


def bucket_averages(
    bucket_idx: npt.NDArray[np.integer],
    x: np.ndarray,
    num_buckets: int
) -> np.ndarray:
    """
    Return the average per bucket in x specified by bucket_idx.
    
    Emtpy buckets will result in np.nan.
    """

    bucket_totals = np.bincount(bucket_idx, weights=x, minlength=num_buckets)
    bucket_counts = np.bincount(bucket_idx, minlength=num_buckets)

    bucket_averages = np.full(num_buckets, np.nan)
    nonzero = bucket_counts > 0
    bucket_averages[nonzero] = (
        bucket_totals[nonzero] / bucket_counts[nonzero]
    )
    return bucket_averages
