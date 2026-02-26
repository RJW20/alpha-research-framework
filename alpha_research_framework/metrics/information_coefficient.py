from typing import override

import alpha_research_framework.market_data as md
from alpha_research_framework.metrics.single_value_metric import (
    SingleValueMetric,
)
from alpha_research_framework.metrics.stats import spearman_rank


class InformationCoefficient(SingleValueMetric):

    ID = "information_coefficient"
 
    @classmethod
    @override
    def compute(cls, signal: md.Array, future_returns: md.Array) -> float:
        """
        Return the Spearman's rank correlation coefficient between `signal` and
        `future_returns`.
        """
        return spearman_rank(signal, future_returns)
