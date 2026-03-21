from typing import TypeAlias

import numpy as np

import alpha_research_framework.observables as observables

Array: TypeAlias = np.memmap
"""
Format for all stores of market-like data (observables and features).

Uses disk-backed storage in the form of a `NumPy` `memmap`. One observable or
feature is stored per `Array`.

`dtype` = `Scalar` \\
`shape` = `(T, N)` where:
- `T` = number of trading days
- `N` = number of tickers
"""

MarketData: TypeAlias = dict[type[observables.Observable], Array]
"""
3D representation of the stock market.

Implemented as a mapping of `Observable` to `Array`.
"""
