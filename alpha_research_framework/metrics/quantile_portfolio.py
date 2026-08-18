from typing import override

import numpy as np
import numpy.typing as npt

import alpha_research_framework.cross_section as xs

from . import stats
from .multi_value_metric import MultiValueMetric


class QuantilePortfolio(MultiValueMetric):

    __quantiles__ = 10

    ID = "quantile_portfolio"

    MEASURES = [f"Q{i+1}" for i in range(__quantiles__)]
    
    @classmethod
    @override
    def compute(
        cls,
        signal: xs.Array,
        forward_returns: xs.Array
    ) -> npt.NDArray[np.floating]:
        """
        Return a `NumPy` array of average `forward_returns` per quantile in
        `signal`.
        """

        valid = stats.extract_valid(signal, forward_returns)
        if not valid:
            return np.full(cls.__quantiles__, np.nan)
        signal_clean, forward_returns_clean = valid
        
        q_idx = stats.quantile_indices(signal_clean, cls.__quantiles__)
        q_avg = stats.bucket_averages(
            q_idx,
            forward_returns_clean,
            cls.__quantiles__,
        )
        return q_avg
