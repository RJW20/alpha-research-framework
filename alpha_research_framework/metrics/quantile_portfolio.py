from typing import override

import numpy as np
import numpy.typing as npt

import alpha_research_framework.market_data as md
import alpha_research_framework.metrics.stats as stats
from alpha_research_framework.metrics.multi_value_metric import (
    MultiValueMetric,
)


class QuantilePorfolio(MultiValueMetric):

    __quantiles__ = 10

    ID = "quantile_portfolio"

    MEASURE_GROUP = "quantiles"
    MEASURES = [f"Q{i+1}" for i in range(__quantiles__)]
    
    @classmethod
    @override
    def compute(
        cls,
        signal: md.Array,
        future_returns: md.Array
    ) -> npt.NDArray[np.floating]:
        """
        Return a `NumPy` array of average `future_returns` per quantile in
        `signal`.
        """

        valid = stats.extract_valid(signal, future_returns)
        if not valid:
            return np.full(cls.__quantiles__, np.nan)
        signal_clean, future_returns_clean = valid
        
        q_idx = stats.quantile_indices(signal_clean, cls.__quantiles__)
        q_avg = stats.bucket_averages(
            q_idx,
            future_returns_clean,
            cls.__quantiles__,
        )
        return q_avg
