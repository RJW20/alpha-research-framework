from abc import ABC, abstractmethod
from typing import ParamSpec

import numpy as np
from numba import njit

from alpha_research_framework.features.feature_error import FeatureError
from alpha_research_framework.features.feature_spec import FeatureSpec
from alpha_research_framework.features.feature_tag import FeatureTag
from alpha_research_framework.features.features import Features
from alpha_research_framework.market_data_view import (
    MarketArray,
    MarketDataView,
)


class Feature(ABC):
    """Abstract feature with automatic dependency checking."""

    TAG: FeatureTag

    def __init__(self) -> None:
        self._name: str
        self._dependencies: set[FeatureSpec] = set()

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        cls._wrap_init()
        cls._wrap_compute()

    @property
    def name(self) -> str:
        return self._name
    
    @property
    def dependencies(self) -> set[FeatureSpec]:
        return self._dependencies

    @abstractmethod
    def compute(
        self,
        market_data: MarketDataView,
        features: Features,
        out: MarketArray
    ) -> None:
        """
        Populate out with values calculated from raw market data and/or
        already computed features.
        """
        ...

    @classmethod
    def _wrap_init(cls) -> None:
        """
        Wrap the __init__ method with a check to ensure subclasses cannot depend
        on features with a higher TAG.
        """

        original = cls.__init__

        P = ParamSpec("P")
        def wrapped(self: Feature, *args: P.args, **kwargs: P.kwargs) -> None:
            original(self, *args, **kwargs)
            for feature in self._dependencies:
                if feature.tag > self.TAG:
                    raise FeatureError(
                        f"Feature {self._name} cannot be instantiated: it "
                        f"cannot depend on {feature.name} with higher TAG."
                    )

        cls.__init__ = wrapped

    @classmethod
    def _wrap_compute(cls) -> None:
        """Ensure all dependencies are present in the given features."""

        original = getattr(cls, "compute", None)
        if original is None or getattr(original, "__isabstractmethod__", False):
            return

        def wrapper(
            self: Feature,
            market_data: MarketDataView,
            features: Features,
            out: MarketArray
        ) -> None:
            missing = self._dependencies - features.keys()
            if missing:
                raise FeatureError(
                    f"Feature {self._name} cannot be computed: missing "
                    f"dependencies {missing}."
                )
            return original(self, market_data, features, out)

        setattr(cls, "compute", wrapper)

    @staticmethod
    @njit
    def _rolling_std(
        values: MarketArray,
        lookback: int,
        out: MarketArray
    ) -> None:
        """
        Populate out with the rolling standard deviation of values over the
        given lookback period.
        
        Calculated with Bessel correction, ignoring NaNs.
        Implemented with memory-efficient streaming.
        Optimised for Numba.
        """

        T, N = values.shape
        out[:lookback - 1] = np.nan

        s = np.zeros(N, dtype=np.float64)
        s2 = np.zeros(N, dtype=np.float64)
        obs = np.zeros(N, dtype=np.int64)

        # First windows < lookback
        for t in range(lookback):

            vt = values[t]
            for j in range(N):
                v = vt[j]
                if not np.isnan(v):
                    s[j] += v
                    s2[j] += v * v
                    obs[j] += 1

            for j in range(N):
                n = obs[j]
                if n > 1:
                    out[t, j] = np.sqrt((s2[j] - (s[j] * s[j]) / n) / (n - 1))
                else:
                    out[t, j] = np.nan

        # Rolling updates
        for t in range(lookback, T):
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
