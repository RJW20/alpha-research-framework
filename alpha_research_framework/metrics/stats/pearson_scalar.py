import numpy as np
from numba import njit


@njit
def _dot_product(x: np.ndarray, y: np.ndarray) -> float:
    """
    Return the dot product of x and y.
    
    Assumes x and y to be of the same shape.
    """
    sum = 0
    for i in range(len(x)):
        sum += x[i] * y[i]
    return sum


@njit
def pearson_scalar(x: np.ndarray, y: np.ndarray) -> float:
    """
    Return the Pearson's correlation coefficient between x and y.
    
    Assumes x and y to be of the same shape.
    """

    xc, yc = x - x.mean(), y - y.mean()
    numerator = _dot_product(xc, yc)
    denominator = np.sqrt(_dot_product(xc, xc) * _dot_product(yc, yc))
    if denominator != 0.0:
        return numerator / denominator
    else:
        return np.nan
