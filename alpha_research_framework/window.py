from enum import Enum


class Window(Enum):
    """Enum defining the number of trading days within common time-periods."""

    DAY = 1
    WEEK = 5
    MONTH = 20
    QUARTER = 63
    HALF_YEAR = 126
    YEAR = 252
