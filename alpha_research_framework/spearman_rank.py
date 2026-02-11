import numpy as np
import numpy.typing as npt


def rankdata(x: np.ndarray) -> npt.NDArray[np.float32]:
    """
    Return an array containing the rank for each element in x.
    
    Handles ties by averaging ranks.
    """

    sorter = np.argsort(x, kind='mergesort')
    sorted_x = x[sorter]

    _, inverse, counts = np.unique(
        sorted_x,
        return_inverse=True,
        return_counts=True
    )
    cum_counts = np.cumsum(counts)
    start_positions = cum_counts - counts
    avg_ranks = (start_positions + cum_counts - 1) / 2

    ranks_sorted = avg_ranks[inverse].astype(np.float32)
    ranks = np.empty_like(ranks_sorted, dtype=np.float32)
    ranks[sorter] = ranks_sorted

    return ranks

def pearson_scalar(x: np.ndarray, y: np.ndarray) -> float:
    """Return the Pearson's correlation coefficient between x and y."""

    xm = x.mean()
    ym = y.mean()
    numerator = np.dot(x - xm, y - ym)
    denominator = np.sqrt(np.dot(x - xm, x - xm) * np.dot(y - ym, y - ym))
    return numerator / denominator

def spearman_rank(x: np.ndarray, y: np.ndarray) -> float:
    """
    Return the Spearman's rank correlation coefficient between x and y.
    
    Omits data that is np.nan in either x or y from the calculation.
    """

    mask = ~np.isnan(x) & ~np.isnan(y)
    x_clean, y_clean = x[mask], y[mask]
    x_rank, y_rank = rankdata(x_clean), rankdata(y_clean)
    return pearson_scalar(x_rank, y_rank)
