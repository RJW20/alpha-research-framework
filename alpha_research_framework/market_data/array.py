from typing import TypeAlias

import numpy as np

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
