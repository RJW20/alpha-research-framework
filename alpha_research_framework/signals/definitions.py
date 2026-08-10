import alpha_research_framework.series as series

from .series_signal import SeriesSignal

# ---------------------------------- Notes -------------------------------------
# Contained here is a full definition of all built-in Signals that this library
# provides.
# For information on creating custom Signals see the "Signals" section of the
# documentation.
# ---------------------------------- Notes -------------------------------------

# ------------------------------------------------------------------------------
# Observables
# ------------------------------------------------------------------------------

class Open(SeriesSignal):
    SERIES = series.Open

class High(SeriesSignal):
    SERIES = series.High

class Low(SeriesSignal):
    SERIES = series.Low

class Close(SeriesSignal):
    SERIES = series.Close

class Volume(SeriesSignal):
    SERIES = series.Volume

# ------------------------------------------------------------------------------
# Log Observables
# ------------------------------------------------------------------------------

class LogOpen(SeriesSignal):
    SERIES = series.LogOpen

class LogHigh(SeriesSignal):
    SERIES = series.LogHigh

class LogLow(SeriesSignal):
    SERIES = series.LogLow

class LogClose(SeriesSignal):
    SERIES = series.LogClose

class LogVolume(SeriesSignal):
    SERIES = series.LogVolume

# ------------------------------------------------------------------------------
# Lagged Log Observables
# ------------------------------------------------------------------------------

class LogCloseLag1d(SeriesSignal):
    SERIES = series.LogCloseLag1d

class LogCloseLag5d(SeriesSignal):
    SERIES = series.LogCloseLag5d

class LogCloseLag20d(SeriesSignal):
    SERIES = series.LogCloseLag20d

class LogCloseLag63d(SeriesSignal):
    SERIES = series.LogCloseLag63d

class LogCloseLag126d(SeriesSignal):
    SERIES = series.LogCloseLag126d

class LogCloseLag252d(SeriesSignal):
    SERIES = series.LogCloseLag252d

class LogVolumeLag1d(SeriesSignal):
    SERIES = series.LogVolumeLag1d

class LogVolumeLag5d(SeriesSignal):
    SERIES = series.LogVolumeLag5d

class LogVolumeLag20d(SeriesSignal):
    SERIES = series.LogVolumeLag20d

class LogVolumeLag63d(SeriesSignal):
    SERIES = series.LogVolumeLag63d

class LogVolumeLag126d(SeriesSignal):
    SERIES = series.LogVolumeLag126d

class LogVolumeLag252d(SeriesSignal):
    SERIES = series.LogVolumeLag252d

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
# Rolling Averages
# ------------------------------------------------------------------------------

class RollingAvgClose5d(SeriesSignal):
    SERIES = series.RollingAvgClose5d

class RollingAvgClose20d(SeriesSignal):
    SERIES = series.RollingAvgClose20d

class RollingAvgClose63d(SeriesSignal):
    SERIES = series.RollingAvgClose63d

class RollingAvgClose126d(SeriesSignal):
    SERIES = series.RollingAvgClose126d

class RollingAvgClose252d(SeriesSignal):
    SERIES = series.RollingAvgClose252d

class RollingAvgVolume5d(SeriesSignal):
    SERIES = series.RollingAvgVolume5d

class RollingAvgVolume20d(SeriesSignal):
    SERIES = series.RollingAvgVolume20d

class RollingAvgVolume63d(SeriesSignal):
    SERIES = series.RollingAvgVolume63d

class RollingAvgVolume126d(SeriesSignal):
    SERIES = series.RollingAvgVolume126d

class RollingAvgVolume252d(SeriesSignal):
    SERIES = series.RollingAvgVolume252d

# ------------------------------------------------------------------------------
# Rolling Standard Deviations
# ------------------------------------------------------------------------------

class RollingStdClose5d(SeriesSignal):
    SERIES = series.RollingStdClose5d

class RollingStdClose20d(SeriesSignal):
    SERIES = series.RollingStdClose20d

class RollingStdClose63d(SeriesSignal):
    SERIES = series.RollingStdClose63d

class RollingStdClose126d(SeriesSignal):
    SERIES = series.RollingStdClose126d

class RollingStdClose252d(SeriesSignal):
    SERIES = series.RollingStdClose252d

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
