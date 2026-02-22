import numpy as np

from alpha_research_framework.metrics.stats import extract_valid
from alpha_research_framework.metrics.stats.pearson_scalar import pearson_scalar
from alpha_research_framework.metrics.stats.rankdata import rankdata


def spearman_rank(x: np.ndarray, y: np.ndarray) -> float:
    """
    Return the Spearman's rank correlation coefficient between x and y.
    
    Omits data that is np.nan in either x or y from the calculation.
    """

    valid = extract_valid(x, y)
    if valid is None:
        return np.nan
    x_clean, y_clean = valid
    x_rank, y_rank = rankdata(x_clean), rankdata(y_clean)
    return pearson_scalar(x_rank, y_rank)
