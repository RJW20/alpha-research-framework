import timeit
import unittest
from functools import partial
from typing import Any, Callable

import numpy as np
import numpy.typing as npt

import alpha_research_framework.features.transforms as transforms
import alpha_research_framework.market_data as md
import alpha_research_framework.metrics.stats as stats
from alpha_research_framework.scalar import Scalar


class TestNumba(unittest.TestCase):
    """
    Class for testing speed-up of numba.njit optimised functions vs plain numpy
    versions.
    """

    RUNS_PER_FUNC = 100
    T = 5000
    N = 500
    Q = 10

    def _assert_numba_faster(
        self,
        numba_func: Callable[..., Any],
        plain_func: Callable[..., Any],
        *args: Any,
        **kwargs: Any
    ) -> None:
        """
        Assert the average runtime over RUNS_PER_FUNC of numba_func is smaller
        compared to plain_func.
        
        Prints how many times faster.
        """

        # compile numba function first
        numba_func(*args, **kwargs)

        numba_runtime = TestNumba._time_function(
            partial(numba_func, *args, **kwargs)
        )
        plain_runtime = TestNumba._time_function(
            partial(plain_func, *args, **kwargs)
        )
        self.assertLess(numba_runtime, plain_runtime)
        
        speed_up = plain_runtime / numba_runtime
        print(f"{numba_func.__name__} {speed_up:.2f}X faster")

    @staticmethod
    def _time_function(func: Callable[[], Any]) -> float:
        """
        Return the average runtime of func over RUNS_PER_FUNC.
        """
        n = TestNumba.RUNS_PER_FUNC
        return timeit.timeit(func, number=n) / n
    
    def test_rolling_average(self) -> None:

        def plain_rolling_avg(arr: md.Array, lookback: int) -> None:
            
            arr_clean = np.nan_to_num(arr, nan=0.0)
            csum = np.cumsum(arr_clean, axis=0)
            rolling_sum = np.empty_like(csum)
            rolling_sum[:lookback] = csum[:lookback]
            rolling_sum[lookback:] = csum[lookback:] - csum[:-lookback]

            valid = ~np.isnan(arr)
            ccount = np.cumsum(valid, axis=0)
            rolling_count = np.empty_like(ccount)
            rolling_count[:lookback] = ccount[:lookback]
            rolling_count[lookback:] = ccount[lookback:] - ccount[:-lookback]

            arr[:] = rolling_sum / rolling_count

        rng = np.random.default_rng(0)
        arr = rng.uniform(0, 10, (TestNumba.T, TestNumba.N)).astype(Scalar)
        lookback = 20
        self._assert_numba_faster(
            transforms.RollingAvg._rolling_avg,
            plain_rolling_avg,
            arr,
            lookback,
        )

    def test_rolling_std(self) -> None:

        def plain_rolling_std(arr: md.Array, lookback: int) -> None:
            
            arr_clean = np.nan_to_num(arr, nan=0.0)

            csum = np.cumsum(arr_clean, axis=0)
            rolling_sum = np.empty_like(csum, dtype=np.float64)
            rolling_sum[:lookback] = csum[:lookback]
            rolling_sum[lookback:] = csum[lookback:] - csum[:-lookback]

            csum2 = np.cumsum(np.square(arr_clean), axis=0)
            rolling_sum2 = np.empty_like(csum2, dtype=np.float64)
            rolling_sum2[:lookback] = csum2[:lookback]
            rolling_sum2[lookback:] = csum2[lookback:] - csum2[:-lookback]

            valid = ~np.isnan(arr)
            ccount = np.cumsum(valid, axis=0)
            rolling_count = np.empty_like(ccount, dtype=int)
            rolling_count[:lookback] = ccount[:lookback]
            rolling_count[lookback:] = ccount[lookback:] - ccount[:-lookback]

            arr[:] = np.sqrt((rolling_sum2 - np.square(rolling_sum) / rolling_count) / (rolling_count - 1))
            arr[rolling_count < 2] = np.nan

        rng = np.random.default_rng(0)
        arr = rng.uniform(0, 10, (TestNumba.T, TestNumba.N)).astype(Scalar)
        lookback = 20
        self._assert_numba_faster(
            transforms.RollingStd._rolling_std,
            plain_rolling_std,
            arr,
            lookback,
        )

    def test_pearson_scalar(self) -> None:
            
        def plain_pearson_scalar(x: np.ndarray, y: np.ndarray) -> float:
            xc, yc = x - x.mean(), y - y.mean()
            numerator = np.dot(xc, yc)
            denominator = np.sqrt(np.dot(xc, xc) * np.dot(yc, yc))
            if denominator != 0.0:
                return numerator / denominator
            else:
                return np.nan
            
        rng = np.random.default_rng(0)
        x = rng.uniform(0, TestNumba.N, TestNumba.N)
        y = rng.uniform(0, TestNumba.N, TestNumba.N)
        self._assert_numba_faster(
            stats.pearson_scalar,
            plain_pearson_scalar,
            x,
            y,
        )

    def test_quantile_indices(self) -> None:
        
        def plain_quantile_indices(
            x: np.ndarray,
            num_quantiles: int
        ) -> npt.NDArray[np.integer]:
            N = len(x)
            order = np.argsort(x)
            q_idx = np.empty_like(order)
            q_idx[order] = np.floor(
                np.arange(N) * num_quantiles / N
            ).astype(int)
            return q_idx
        
        rng = np.random.default_rng(0)
        x = rng.integers(0, TestNumba.N, TestNumba.N)
        self._assert_numba_faster(
            stats.quantile_indices,
            plain_quantile_indices,
            x,
            TestNumba.Q,
        )
        
    def test_bucket_averages(self) -> None:

        def plain_bucket_averages(
            bucket_idx: npt.NDArray[np.integer],
            x: np.ndarray,
            num_buckets: int
        ) -> np.ndarray:
            bucket_totals = np.bincount(
                bucket_idx,
                weights=x,
                minlength=num_buckets
            )
            bucket_counts = np.bincount(bucket_idx, minlength=num_buckets)
            bucket_averages = np.full(num_buckets, np.nan)
            nonzero = bucket_counts > 0
            bucket_averages[nonzero] = (
                bucket_totals[nonzero] / bucket_counts[nonzero]
            )
            return bucket_averages
        
        rng = np.random.default_rng(0)
        x = rng.integers(0, TestNumba.N, TestNumba.N)
        bucket_idx = stats.quantile_indices(x, TestNumba.Q)
        self._assert_numba_faster(
            stats.bucket_averages,
            plain_bucket_averages,
            bucket_idx,
            x,
            TestNumba.Q,
        )


if __name__ == "__main__":
    unittest.main()
