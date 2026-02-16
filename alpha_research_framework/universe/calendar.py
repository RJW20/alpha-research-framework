import pandas as pd


class Calendar:
    """
    Trading calendar defining the global date index for the universe.

    Provides a fixed, ordered mapping between integer time indices and trading
    dates. All time-dependent arrays in the universe are indexed along this
    calendar.
    """

    def __init__(self, index: pd.Index) -> None:
        self.index = index
        self.T = len(self.index)

    def date(self, t: int) -> pd.Timestamp:
        return self.index[t]
    
    def t(self, date: pd.Timestamp):
        return self.index.get_loc(date)
