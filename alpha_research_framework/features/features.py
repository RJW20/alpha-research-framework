import alpha_research_framework.features.transforms as transforms
import alpha_research_framework.observables as observables
from alpha_research_framework.features.feature import Feature
from alpha_research_framework.features.primitive_feature import PrimitiveFeature
from alpha_research_framework.window import Window

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
