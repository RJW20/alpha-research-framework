from __future__ import annotations

from abc import abstractmethod
from enum import IntEnum
from typing import Any, ClassVar

import numpy as np
from numba import njit

import alpha_research_framework.market_data as md
from alpha_research_framework.class_var_validator import ClassVarValidator
from alpha_research_framework.features.dependency_error import DependencyError
from alpha_research_framework.features.features import Features
from alpha_research_framework.operator import Operator


class Feature(Operator, ClassVarValidator, registry_root=True, abstract=True):
    """
    Abstract base class for cross-sectional features with automatic subclass
    validation and runtime missing dependency error reporting.
    
    Any concrete subclass must define:
    - `ID`: `str` - unique identifier
    - `TAG`: `Feature.Tag` (`PREDICTOR` or `TARGET`) - usage classification
    - `DEPENDENCIES`: `set[Feature]` - prerequisite features this feature needs
    to compute
    - `compute(
        cls,
        market_data: md.MarketData,
        features: Features,
        out: md.Array,
    ) -> None:` - classmethod for calculating the feature
    """

    class Tag(IntEnum):
        PREDICTOR = 0
        TARGET = 1

    TAG: ClassVar[Tag]
    DEPENDENCIES: ClassVar[set[type[Feature]]]

    def __init_subclass__(cls, abstract: bool = False, **kwargs: Any) -> None:
        """
        Initialise a new subclass.

        If `abstract=False` asserts definition and type of `TAG`, definition,
        type and `TAG` of `DEPENDENCIES`, and wraps `compute` with error
        reporting for incomplete `features` argument.
        """

        kwargs["abstract"] = abstract
        super().__init_subclass__(**kwargs)

        if abstract:
            return

        cls.assert_class_var(name="TAG", type=Feature.Tag)

        cls.assert_class_var_container(
            name="DEPENDENCIES",
            container_type=set,
            element_type=type,
        )
        cls._assert_dependencies_tags()

        cls._wrap_compute()

    @classmethod
    @abstractmethod
    def compute(
        cls,
        market_data: md.MarketData,
        features: Features,
        out: md.Array,
    ) -> None:
        """
        Populate `out` with values calculated from raw `market_data` and/or
        already computed `features`.
        """
        ...

    @classmethod
    def _assert_dependencies_tags(cls) -> None:
        """
        Assert that all features in `cls.DEPENDENCIES` do not have a higher
        `TAG` than `cls`.
        """

        for feature in cls.DEPENDENCIES:
            if feature.TAG > cls.TAG:
                raise ValueError(
                    f"Feature {cls.__name__} cannot depend on "
                    f"{feature.__class__.__name__} with higher TAG"
                )

    @classmethod
    def _wrap_compute(cls) -> None:
        """
        Wrap `compute` with error reporting for incomplete `features` argument.
        """

        original = getattr(cls, "compute")
        if getattr(original, "__isabstractmethod__", False):
            return

        def wrapper(
            cls: type[Feature],
            market_data: md.MarketData,
            features: Features,
            out: md.Array,
        ) -> None:
            try:
                original(market_data, features, out)
            except KeyError as e:
                missing_dependency = cls.from_id(e.args[0])
                raise DependencyError(
                    f"Feature {cls.__name__} cannot be computed: missing "
                    f"dependency {missing_dependency.__name__}"
                )

        setattr(cls, "compute", classmethod(wrapper))

    @staticmethod
    @njit
    def _rolling_std(values: md.Array,  lookback: int, out: md.Array) -> None:
        """
        Populate `out` with the rolling standard deviation of `values` over the
        `lookback` period.
        
        Calculated with Bessel correction, ignoring NaNs.
        Implemented with memory-efficient streaming.
        Optimised for Numba.
        """

        T, N = values.shape
        s = np.zeros(N, dtype=np.float64)
        s2 = np.zeros(N, dtype=np.float64)
        obs = np.zeros(N, dtype=np.int64)

        # Account for lookback larger than T
        if lookback > T:
            windows_lt_lookback = T
        else:
            windows_lt_lookback = lookback

        # First windows < lookback
        for t in range(windows_lt_lookback):

            vt = values[t]
            for j in range(N):
                
                v = vt[j]
                if not np.isnan(v):
                    s[j] += v
                    s2[j] += v * v
                    obs[j] += 1

                n = obs[j]
                if n > 1:
                    out[t, j] = np.sqrt((s2[j] - (s[j] * s[j]) / n) / (n - 1))
                else:
                    out[t, j] = np.nan

        # Rolling updates
        for t in range(windows_lt_lookback, T):
            vt_old = values[t - lookback]
            vt_new = values[t]

            for j in range(N):

                v_old = vt_old[j]
                if not np.isnan(v_old):
                    s[j] -= v_old
                    s2[j] -= v_old * v_old
                    obs[j] -= 1

                v_new = vt_new[j]
                if not np.isnan(v_new):
                    s[j] += v_new
                    s2[j] += v_new * v_new
                    obs[j] += 1

                n = obs[j]
                if n > 1:
                    out[t, j] = np.sqrt((s2[j] - (s[j] * s[j]) / n) / (n - 1))
                else:
                    out[t, j] = np.nan
