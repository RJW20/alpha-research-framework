from typing import TypeAlias

import pandas as pd

TickerData: TypeAlias = pd.DataFrame
"""
`Pandas` `DataFrame` indexed by trading date containing raw market data for a
ticker.

The included data is `"open"`, `"high"`, `"low"`, `"close"`,
`"adj_close"`, `"volume"`, `"adj_factor"`.
"""
