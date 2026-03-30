from abc import abstractmethod
from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt

import alpha_research_framework.cross_section as xs
from alpha_research_framework.class_var_validator import ClassVarValidator

from .metric import Metric


class MultiValueMetric(Metric, ClassVarValidator, register=False):
    """
    Abstract base class for multi-valued metrics measuring the performance of
    cross-sectional alphas with automatic subclass validation.

    Any subclass must define:
    - `ID`: `str` - unique identifier
    - `MEASURES`: `list[str]` - names of each measure
    - `compute(
        cls,
        signal: xs.Array,
        forward_returns: xs.Array,
      ) -> npt.NDArray[np.floating]:` - classmethod for calculating the metric
    """

    MEASURES: ClassVar[list[str]]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Asserts definition, type and value of `MEASURES`."""

        super().__init_subclass__(**kwargs)
        
        cls.assert_class_var_container(
            name="MEASURES",
            container_type=list,
            element_type=str,
            bad_values={""},
        )

    @classmethod
    @abstractmethod
    def compute(
        cls,
        signal: xs.Array,
        forward_returns: xs.Array
    ) -> npt.NDArray[np.floating]:
        """
        Return values containing a measure of correlation between `signal` and
        `forward_returns`.
        """
