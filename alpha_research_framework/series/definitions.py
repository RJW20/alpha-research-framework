# ruff: noqa: E501
import alpha_research_framework.observables as observables
from alpha_research_framework.window import Window

from . import transforms
from .observable_series import ObservableSeries
from .series import Series
from .transform import transform

# ------------------------------------------------------ Notes ---------------------------------------------------------
# Contained here is a full definition of all built-in Series that this library provides.
# For information on creating custom Series see the "Series" section of the documentation.
# ------------------------------------------------------ Notes ---------------------------------------------------------


# ---------------------------------------------------- Predictors ------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# Observables
# ----------------------------------------------------------------------------------------------------------------------

class Open(ObservableSeries):
    """`series` = `observables.Open`"""
    TAG = Series.Tag.PREDICTOR
    OBSERVABLE = observables.Open

class High(ObservableSeries):
    """`series` = `observables.High`"""
    TAG = Series.Tag.PREDICTOR
    OBSERVABLE = observables.High

class Low(ObservableSeries):
    """`series` = `observables.Low`"""
    TAG = Series.Tag.PREDICTOR
    OBSERVABLE = observables.Low

class Close(ObservableSeries):
    """`series` = `observables.Close`"""
    TAG = Series.Tag.PREDICTOR
    OBSERVABLE = observables.Close

class Volume(ObservableSeries):
    """`series` = `observables.Volume`"""
    TAG = Series.Tag.PREDICTOR
    OBSERVABLE = observables.Volume

# ----------------------------------------------------------------------------------------------------------------------
# Returns
# ----------------------------------------------------------------------------------------------------------------------

LogClose = transform(Close, func=transforms.log)

LogCloseLag1d   = transform(LogClose, func=transforms.shift_forward, period=Window.DAY.value      )
LogCloseLag5d   = transform(LogClose, func=transforms.shift_forward, period=Window.WEEK.value     )
LogCloseLag20d  = transform(LogClose, func=transforms.shift_forward, period=Window.MONTH.value    )
LogCloseLag63d  = transform(LogClose, func=transforms.shift_forward, period=Window.QUARTER.value  )
LogCloseLag126d = transform(LogClose, func=transforms.shift_forward, period=Window.HALF_YEAR.value)
LogCloseLag252d = transform(LogClose, func=transforms.shift_forward, period=Window.YEAR.value     )

Returns1d   = LogClose - LogCloseLag1d
Returns5d   = LogClose - LogCloseLag5d
Returns20d  = LogClose - LogCloseLag20d
Returns63d  = LogClose - LogCloseLag63d
Returns126d = LogClose - LogCloseLag126d
Returns252d = LogClose - LogCloseLag252d

# ----------------------------------------------------------------------------------------------------------------------
# Volatility
# ----------------------------------------------------------------------------------------------------------------------

Volatility1d   = transform(Returns1d, func=transforms.rolling_std, lookback=Window.DAY.value      )
Volatility5d   = transform(Returns1d, func=transforms.rolling_std, lookback=Window.WEEK.value     )
Volatility20d  = transform(Returns1d, func=transforms.rolling_std, lookback=Window.MONTH.value    )
Volatility63d  = transform(Returns1d, func=transforms.rolling_std, lookback=Window.QUARTER.value  )
Volatility126d = transform(Returns1d, func=transforms.rolling_std, lookback=Window.HALF_YEAR.value)
Volatility252d = transform(Returns1d, func=transforms.rolling_std, lookback=Window.YEAR.value     )

# ----------------------------------------------------------------------------------------------------------------------
# Lagged Close
# ----------------------------------------------------------------------------------------------------------------------

CloseLag1d = transform(Close, func=transforms.shift_forward, period=Window.DAY)

# ---------------------------------------------------- Predictors ------------------------------------------------------


# ----------------------------------------------------- Targets --------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# Forward Returns
# ----------------------------------------------------------------------------------------------------------------------

ForwardReturns1d   = transform(Returns1d,   target=True, func=transforms.shift_back, period=Window.DAY.value      )
ForwardReturns5d   = transform(Returns5d,   target=True, func=transforms.shift_back, period=Window.WEEK.value     )
ForwardReturns20d  = transform(Returns20d,  target=True, func=transforms.shift_back, period=Window.MONTH.value    )
ForwardReturns63d  = transform(Returns63d,  target=True, func=transforms.shift_back, period=Window.QUARTER.value  )
ForwardReturns126d = transform(Returns126d, target=True, func=transforms.shift_back, period=Window.HALF_YEAR.value)
ForwardReturns252d = transform(Returns252d, target=True, func=transforms.shift_back, period=Window.YEAR.value     )

# ----------------------------------------------------- Targets --------------------------------------------------------
