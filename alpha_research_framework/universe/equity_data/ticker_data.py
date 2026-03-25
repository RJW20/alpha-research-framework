from typing import TypeAlias

import pandas as pd

TickerData: TypeAlias = pd.DataFrame
"""
`Pandas` `DataFrame` indexed by trading date containing raw (adjusted) OHLC and
volume data for a ticker.
"""
