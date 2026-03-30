from typing import Any, override

import numpy as np
from numba import njit

import alpha_research_framework.market_data as md

from .transform import Transform


class RollingAvg(Transform):
    """`feature` -> `sum(feature[t-lookback+1],...,feature[t])/lookback`"""
    
    @classmethod
    @override
    def compute(cls, arr: md.Array, *, lookback: int, **kwargs: Any) -> None:
        RollingAvg._rolling_avg(arr, int(lookback))

    @staticmethod
    @njit
    def _rolling_avg(arr: md.Array, lookback: int) -> None:
        """
        Modify `arr` in-place with its rolling average over `lookback`.

        Implemented with memory-efficient streaming.
        Optimised for Numba.
        """

        T, N = arr.shape
        s = np.zeros(N, dtype=np.float64)
        obs = np.zeros(N, dtype=np.int64)
        buffer = np.empty((lookback, N), dtype=np.float64)

        # Account for lookback larger than T
        windows_lt_lookback = min(lookback, T)

        # First windows < lookback
        for t in range(windows_lt_lookback):

            at = arr[t]
            for j in range(N):
                
                a = at[j]
                if not np.isnan(a):
                    s[j] += a
                    obs[j] += 1

                n = obs[j]
                if n:
                    buffer[t, j] = s[j] / n
                else:
                    buffer[t, j] = np.nan

        # Rolling updates
        at_old = np.empty_like(s) # needs its own memory
        for t in range(windows_lt_lookback, T):

            # Read values for updating
            at_old[:] = arr[t - lookback]
            at_new = arr[t]

            # Flush line of buffer
            arr[t - lookback] = buffer[t % lookback]

            for j in range(N):

                a_old = at_old[j]
                if not np.isnan(a_old):
                    s[j] -= a_old
                    obs[j] -= 1

                a_new = at_new[j]
                if not np.isnan(a_new):
                    s[j] += a_new
                    obs[j] += 1

                n = obs[j]
                if n:
                    buffer[t % lookback, j] = s[j] / n
                else:
                    buffer[t % lookback, j] = np.nan

        # Flush remaining buffer
        start = max(0, T - lookback)
        for t in range(start, T):
            arr[t] = buffer[t % lookback]
