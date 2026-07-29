from typing import Any, override

import numpy as np

import alpha_research_framework.market_data as md

from .transform import Transform


class Log(Transform):
    """`feature` -> `log(feature)`"""
    
    @classmethod
    @override
    def compute(cls, arr: md.Array, **kwargs: Any) -> None:
        arr[:] = np.log(arr)
