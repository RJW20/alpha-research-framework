from typing import Any, override

import numpy as np

import alpha_research_framework.market_data as md

from .transform import Transform


class LeadDifference(Transform):
    """`feature` -> `feature[t+lead] - feature[t]`"""

    @classmethod
    @override
    def compute(cls, arr: md.Array, *, lead: int, **kwargs: Any) -> None:
        arr[:-lead] = arr[lead:] - arr[:-lead]
        arr[-lead:] = np.nan
