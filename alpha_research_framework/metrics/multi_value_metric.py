from abc import abstractmethod
from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt

import alpha_research_framework.market_data as md
from alpha_research_framework.class_var_validator import ClassVarValidator
from alpha_research_framework.metrics.metric import Metric


class MultiValueMetric(Metric, ClassVarValidator, abstract=True):
    """
    Abstract base class for multi-valued metrics measuring the performance of
    cross-sectional alphas with automatic subclass validation.

    Any subclass must define:
    - `ID`: `str` - unique identifier
    - `MEASURE_GROUP`: `str` - umbrella term for the measures within the metric
    - `MEASURES`: `list[str]` - names of each measure
    - `compute(
        cls,
        signal: md.Array,
        future_returns: md.Array,
      ) -> npt.NDArray[np.floating]:` - classmethod for calculating the metric
    """

    MEASURE_GROUP: ClassVar[str]
    MEASURES: ClassVar[list[str]]

    def __init_subclass__(cls, abstract: bool = False, **kwargs: Any) -> None:
        """
        If `abstract=False` asserts definition, type and value of
        `MEASURE_GROUP` and `MEASURES`.
        """

        kwargs["abstract"] = abstract
        super().__init_subclass__(**kwargs)

        if abstract:
            return
        
        cls.assert_class_var(name="MEASURE_GROUP", type=str, bad_values={""})
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
        signal: md.Array,
        future_returns: md.Array
    ) -> npt.NDArray[np.floating]:
        """
        Return values containing a measure of correlation between `signal` and
        `future_returns`.
        """
