import pandas as pd
import pandas_market_calendars as mcal


class Calendar:
    """
    Trading calendar defining the global date index for the universe.

    Provides a fixed, ordered mapping between integer time indices and trading
    dates. All time-dependent arrays in the universe are indexed along this
    calendar.
    """

    def __init__(self, start_date: str, end_date: str) -> None:
        nyse = mcal.get_calendar('NYSE')
        schedule = nyse.schedule(start_date, end_date)
        self.index = schedule.index.astype('datetime64[ms]')
        self.T: int = len(self.index)

    def date(self, t: int) -> pd.Timestamp:
        return self.index[t]
