from typing import override

import alpha_research_framework.cross_section as xs

from .single_value_metric import SingleValueMetric
from .stats import spearman_rank


class InformationCoefficient(SingleValueMetric):

    ID = "information_coefficient"
 
    @classmethod
    @override
    def compute(cls, signal: xs.Array, forward_returns: xs.Array) -> float:
        """
        Return the Spearman's rank correlation coefficient between `signal` and
        `forward_returns`.
        """
        return spearman_rank(signal, forward_returns)
