from abc import abstractmethod

import numpy as np
import numpy.typing as npt

import alpha_research_framework.market_data as md
from alpha_research_framework.operator import Operator


class Metric(Operator, registry_root=True, abstract=True):
    """
    Abstract base class for metrics measuring the performance of cross-sectional
    alphas with automatic subclass validation.

    Any subclass must define:
    - `ID`: `str` - unique identifier
    - `compute(
        cls,
        signal: md.Array,
        future_returns: md.Array,
      ) -> float | npt.NDArray[np.floating]:` - classmethod for calculating the
    metric
    """

    @classmethod
    @abstractmethod
    def compute(
        cls,
        signal: md.Array,
        future_returns: md.Array
    ) -> float | npt.NDArray[np.floating]:
        """
        Return a value or values containing a measure of correlation between
        `signal` and `future_returns`.
        """
