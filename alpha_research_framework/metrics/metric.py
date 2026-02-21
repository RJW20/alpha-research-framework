from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

import alpha_research_framework.market_data as md
from alpha_research_framework.metrics.metric_error import MetricError


class Metric(ABC):
    """
    Abstract base class for metrics measuring the performance of cross-sectional
    alphas with automatic subclass validation and registry management.

    Any subclass must define:
    - ID: str - unique identifier which can be used to fetch the subclass via
    Metric.from_id
    - dataframe(
        index: pd.Index,
        base_columns: list[tuple[Any,...]],
        base_column_names: list[str]
      ) -> pd.DataFrame - dataframe to store the metric in
    - compute(
        signal: md.Array,
        future_returns: md.Array
      ) -> float | npt.NDArray[np.floating] - value or values measuring
      correlation between signal and future returns
    """

    __registry__: dict[str, type["Metric"]] = dict()

    ID: str | None = None

    def __init_subclass__(cls) -> None:
        """
        Validate definition, type and value of ID and add subclass to registry.
        """

        super().__init_subclass__()

        if cls is Metric:
            return
        
        if cls.ID is None:
            raise MetricError(f"{cls.__name__} must define ID.")
        if not isinstance(cls.ID, str):
            raise TypeError(f"{cls.__name__}.ID must be of type str.")
        if not cls.ID:
            raise ValueError(f"{cls.__name__}.ID cannot be empty.")
        if cls.ID in Metric.__registry__:
            raise MetricError(
                f"{cls.__name__}.ID must be unique (metric with ID '{cls.ID}' "
                "already exists)."
            )

        Metric.__registry__[cls.ID] = cls

    @staticmethod
    def from_id(id: str) -> type["Metric"]:
        """Return the metric with ID = id."""
        if id not in Metric.__registry__:
            raise MetricError(f"Metric with ID '{id}' does not exist.")
        return Metric.__registry__[id]
        
    @staticmethod
    @abstractmethod
    def dataframe(
        index: pd.Index,
        base_columns: list[tuple[Any,...]],
        base_column_names: list[str]
    ) -> pd.DataFrame:
        """
        Return a pd.Dataframe in the format for the metric to store its results.
        """

    @staticmethod
    @abstractmethod
    def compute(
        signal: md.Array,
        future_returns: md.Array
    ) -> float | npt.NDArray[np.floating]:
        """
        Return a value or values containing a measure of correlation between the
        signal and future_returns.
        """
