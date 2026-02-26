from abc import abstractmethod

import alpha_research_framework.market_data as md
from alpha_research_framework.metrics.metric import Metric


class SingleValueMetric(Metric, abstract=True):
    """
    Abstract base class for single-valued metrics measuring the performance of
    cross-sectional alphas with automatic subclass validation.

    Any subclass must define:
    - `ID`: `str` - unique identifier
    - `compute(cls, signal: md.Array, future_returns: md.Array) -> float:` -
    classmethod for calculating the metric
    """

    @classmethod
    @abstractmethod
    def compute(cls, signal: md.Array, future_returns: md.Array) -> float:
        """
        Return a value containing a measure of correlation between `signal` and
        `future_returns`.
        """
