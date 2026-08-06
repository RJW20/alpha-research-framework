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

LogCloseLag1d   = transform(LogClose, func=transforms.shift_forward, period=Window.DAY      )
LogCloseLag5d   = transform(LogClose, func=transforms.shift_forward, period=Window.WEEK     )
LogCloseLag20d  = transform(LogClose, func=transforms.shift_forward, period=Window.MONTH    )
LogCloseLag63d  = transform(LogClose, func=transforms.shift_forward, period=Window.QUARTER  )
LogCloseLag126d = transform(LogClose, func=transforms.shift_forward, period=Window.HALF_YEAR)
LogCloseLag252d = transform(LogClose, func=transforms.shift_forward, period=Window.YEAR     )

Returns1d   = LogClose - LogCloseLag1d
Returns5d   = LogClose - LogCloseLag5d
Returns20d  = LogClose - LogCloseLag20d
Returns63d  = LogClose - LogCloseLag63d
Returns126d = LogClose - LogCloseLag126d
Returns252d = LogClose - LogCloseLag252d

# ----------------------------------------------------------------------------------------------------------------------
# Volatility
# ----------------------------------------------------------------------------------------------------------------------

Volatility1d   = transform(Returns1d, func=transforms.rolling_std, lookback=Window.DAY      )
Volatility5d   = transform(Returns1d, func=transforms.rolling_std, lookback=Window.WEEK     )
Volatility20d  = transform(Returns1d, func=transforms.rolling_std, lookback=Window.MONTH    )
Volatility63d  = transform(Returns1d, func=transforms.rolling_std, lookback=Window.QUARTER  )
Volatility126d = transform(Returns1d, func=transforms.rolling_std, lookback=Window.HALF_YEAR)
Volatility252d = transform(Returns1d, func=transforms.rolling_std, lookback=Window.YEAR     )

# ----------------------------------------------------------------------------------------------------------------------
# Lagged Close
# ----------------------------------------------------------------------------------------------------------------------

CloseLag1d = transform(Close, func=transforms.shift_forward, period=Window.DAY)

# ---------------------------------------------------- Predictors ------------------------------------------------------


# ----------------------------------------------------- Targets --------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# Forward Returns
# ----------------------------------------------------------------------------------------------------------------------

ForwardReturns1d   = transform(Returns1d,   target=True, func=transforms.shift_back, period=Window.DAY      )
ForwardReturns5d   = transform(Returns5d,   target=True, func=transforms.shift_back, period=Window.WEEK     )
ForwardReturns20d  = transform(Returns20d,  target=True, func=transforms.shift_back, period=Window.MONTH    )
ForwardReturns63d  = transform(Returns63d,  target=True, func=transforms.shift_back, period=Window.QUARTER  )
ForwardReturns126d = transform(Returns126d, target=True, func=transforms.shift_back, period=Window.HALF_YEAR)
ForwardReturns252d = transform(Returns252d, target=True, func=transforms.shift_back, period=Window.YEAR     )

# ----------------------------------------------------- Targets --------------------------------------------------------
