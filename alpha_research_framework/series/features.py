import alpha_research_framework.observables as observables
from alpha_research_framework.window import Window

from . import transforms
from .feature import Feature
from .primitive_feature import PrimitiveFeature

# ------------------------------------------------------------------------------
# Primitives
# ------------------------------------------------------------------------------

class Price(PrimitiveFeature):
    """`feature` = `observables.Close`"""
    TAG = Feature.Tag.PREDICTOR
    OBSERVABLE = observables.Close

class Volume(PrimitiveFeature):
    """`feature` = `observables.Volume`"""
    TAG = Feature.Tag.PREDICTOR
    OBSERVABLE = observables.Volume

# ------------------------------------------------------------------------------
# Derivatives
# ------------------------------------------------------------------------------

LogPrice = transforms.Log(Price)

Returns1d = transforms.LaggedDifference(LogPrice, lag=Window.DAY)
Returns5d = transforms.LaggedDifference(LogPrice, lag=Window.WEEK)
Returns20d = transforms.LaggedDifference(LogPrice, lag=Window.MONTH)
Returns63d = transforms.LaggedDifference(LogPrice, lag=Window.QUARTER)
Returns126d = transforms.LaggedDifference(LogPrice, lag=Window.HALF_YEAR)
Returns252d = transforms.LaggedDifference(LogPrice, lag=Window.YEAR)

Volatility1d = transforms.RollingStd(Returns1d, lookback=Window.DAY)
Volatility5d = transforms.RollingStd(Returns1d, lookback=Window.WEEK)
Volatility20d = transforms.RollingStd(Returns1d, lookback=Window.MONTH)
Volatility63d = transforms.RollingStd(Returns1d, lookback=Window.QUARTER)
Volatility126d = transforms.RollingStd(Returns1d, lookback=Window.HALF_YEAR)
Volatility252d = transforms.RollingStd(Returns1d, lookback=Window.YEAR)

# ------------------------------------------------------------------------------
# Target Derivatives
# ------------------------------------------------------------------------------

ForwardReturns1d = transforms.LeadDifference(
    LogPrice,
    target=True,
    lead=Window.DAY,
)
ForwardReturns5d = transforms.LeadDifference(
    LogPrice,
    target=True,
    lead=Window.WEEK,
)
ForwardReturns20d = transforms.LeadDifference(
    LogPrice,
    target=True,
    lead=Window.MONTH,
)
ForwardReturns63d = transforms.LeadDifference(
    LogPrice,
    target=True,
    lead=Window.QUARTER,
)
ForwardReturns126d = transforms.LeadDifference(
    LogPrice,
    target=True,
    lead=Window.HALF_YEAR,
)
ForwardReturns252d = transforms.LeadDifference(
    LogPrice,
    target=True,
    lead=Window.YEAR,
)
