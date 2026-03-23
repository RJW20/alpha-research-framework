import alpha_research_framework.observables as observables
from alpha_research_framework.features.feature import Feature
from alpha_research_framework.features.primitive_feature import PrimitiveFeature

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
