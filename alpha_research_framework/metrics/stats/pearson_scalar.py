import numpy as np


def pearson_scalar(x: np.ndarray, y: np.ndarray) -> float:
    """Return the Pearson's correlation coefficient between x and y."""

    xm = x.mean()
    ym = y.mean()
    numerator = np.dot(x - xm, y - ym)
    denominator = np.sqrt(np.dot(x - xm, x - xm) * np.dot(y - ym, y - ym))
    if denominator != 0:
        return numerator / denominator
    else:
        return np.nan
