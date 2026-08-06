from typing import TypeAlias

import numpy.typing as npt

import alpha_research_framework.series as series
from alpha_research_framework.scalar import Scalar

Array: TypeAlias = npt.NDArray[Scalar]
"""
Format for all stores of cross-sectional data (series, factors and alphas).

One series, factor or alpha is stored per `Array`.

`dtype` = `Scalar` \\
`shape` = `(N,)` where:
- `N` = number of tickers
"""

CrossSection: TypeAlias = dict[type[series.Series], Array]
"""
2D snapshot of a `Universe` at a single timestamp.

Implemented as a mapping of `Series` to `Array`.
"""
