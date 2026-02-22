import timeit
import unittest
from functools import partial
from typing import Any, Callable

import numpy as np
import numpy.typing as npt

import alpha_research_framework.metrics.stats as stats


class TestNumba(unittest.TestCase):
    """
    Class for testing speed-up of numba.njit optimised functions vs plain numpy
    versions.
    """

    RUNS_PER_FUNC = 100
    N = 500
    Q = 10

    def _assert_numba_faster(
        self,
        numba_func: Callable[[Any], Any],
        plain_func: Callable[[Any], Any],
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
        x = rng.uniform(0, self.N, self.N)
        y = rng.uniform(0, self.N, self.N)
        self._assert_numba_faster(
            stats.pearson_scalar,
            plain_pearson_scalar,
            x,
            y
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
            TestNumba.Q
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
            TestNumba.Q
        )


if __name__ == "__main__":
    unittest.main()
