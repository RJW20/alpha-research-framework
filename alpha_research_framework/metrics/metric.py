from abc import abstractmethod

import numpy as np
import numpy.typing as npt

import alpha_research_framework.cross_section as xs
from alpha_research_framework.operator import Operator
from alpha_research_framework.registrable import Registrable


class Metric(Operator, Registrable, registry_root=True):
    """
    Abstract base class for metrics measuring the performance of cross-sectional
    alphas with automatic subclass validation.

    Any subclass must define:
    - `ID`: `str` - unique identifier
    - `compute(cls, xs.Array, xs.Array) -> float | npt.NDArray[np.floating]:` -
    classmethod for calculating the metric
    """

    @classmethod
    @abstractmethod
    def compute(
        cls,
        signal: xs.Array,
        future_returns: xs.Array
    ) -> float | npt.NDArray[np.floating]:
        """
        Return a value or values containing a measure of correlation between
        `signal` and `future_returns`.
        """
