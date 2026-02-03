import pandas as pd


class Calendar:
    """
    Trading calendar defining the global date index for the universe.

    Provides a fixed, ordered mapping between integer time indices and trading
    dates. All time-dependent arrays in the universe are indexed along this
    calendar.
    """

    def __init__(self, start_date: str, end_date: str) -> None:
        self.index = pd.date_range(
            start_date,
            end_date,
            freq='B'
        ).astype('datetime64[ms]')
        self.T: int = len(self.index)

    def date(self, t: int) -> pd.Timestamp:
        return self.index[t]
