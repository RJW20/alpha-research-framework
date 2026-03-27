from typing import Any, override

import numpy as np

import alpha_research_framework.market_data as md

from .transform import Transform


class LaggedDifference(Transform):
    """`feature` -> `feature[t] - feature[t-lag]`"""

    @classmethod
    @override
    def compute(cls, arr: md.Array, *, lag: int, **kwargs: Any) -> None:
        arr[lag:] = arr[lag:] - arr[:-lag]
        arr[:lag] = np.nan
