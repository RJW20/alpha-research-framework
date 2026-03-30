from abc import abstractmethod

import alpha_research_framework.cross_section as xs

from .metric import Metric


class SingleValueMetric(Metric, register=False):
    """
    Abstract base class for single-valued metrics measuring the performance of
    cross-sectional alphas with automatic subclass validation.

    Any subclass must define:
    - `ID`: `str` - unique identifier
    - `compute(cls, signal: xs.Array, forward_returns: xs.Array) -> float:` -
    classmethod for calculating the metric
    """

    @classmethod
    @abstractmethod
    def compute(cls, signal: xs.Array, forward_returns: xs.Array) -> float:
        """
        Return a value containing a measure of correlation between `signal` and
        `forward_returns`.
        """
