import numpy as np


def rankdata(x: np.ndarray) -> np.ndarray:
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

    ranks_sorted = avg_ranks[inverse]
    ranks = np.empty_like(ranks_sorted)
    ranks[sorter] = ranks_sorted

    return ranks
