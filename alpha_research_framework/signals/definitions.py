import alpha_research_framework.series as series

from .series_signal import SeriesSignal

# ---------------------------------- Notes -------------------------------------
# Contained here is a full definition of all built-in Signals that this library
# provides.
# For information on creating custom Signals see the "Signals" section of the
# documentation.
# ---------------------------------- Notes -------------------------------------

# ------------------------------------------------------------------------------
# Returns
# ------------------------------------------------------------------------------

class Returns1d(SeriesSignal):
    SERIES = series.Returns1d

class Returns5d(SeriesSignal):
    SERIES = series.Returns5d

class Returns20d(SeriesSignal):
    SERIES = series.Returns20d

class Returns63d(SeriesSignal):
    SERIES = series.Returns63d

class Returns126d(SeriesSignal):
    SERIES = series.Returns126d

class Returns252d(SeriesSignal):
    SERIES = series.Returns252d

# ------------------------------------------------------------------------------
# Volatility
# ------------------------------------------------------------------------------

class Volatility1d(SeriesSignal):
    SERIES = series.Volatility1d

class Volatility5d(SeriesSignal):
    SERIES = series.Volatility5d

class Volatility20d(SeriesSignal):
    SERIES = series.Volatility20d

class Volatility63d(SeriesSignal):
    SERIES = series.Volatility63d

class Volatility126d(SeriesSignal):
    SERIES = series.Volatility126d

class Volatility252d(SeriesSignal):
    SERIES = series.Volatility252d
