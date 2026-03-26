import alpha_research_framework.features as features

from .primitive_factor import PrimitiveFactor

# ------------------------------------------------------------------------------
# Returns
# ------------------------------------------------------------------------------

class Returns1d(PrimitiveFactor):
    FEATURE = features.Returns1d

class Returns5d(PrimitiveFactor):
    FEATURE = features.Returns5d

class Returns20d(PrimitiveFactor):
    FEATURE = features.Returns20d

class Returns63d(PrimitiveFactor):
    FEATURE = features.Returns63d

class Returns126d(PrimitiveFactor):
    FEATURE = features.Returns126d

class Returns252d(PrimitiveFactor):
    FEATURE = features.Returns252d

# ------------------------------------------------------------------------------
# Volatility
# ------------------------------------------------------------------------------

class Volatility1d(PrimitiveFactor):
    FEATURE = features.Volatility1d

class Volatility5d(PrimitiveFactor):
    FEATURE = features.Volatility5d

class Volatility20d(PrimitiveFactor):
    FEATURE = features.Volatility20d

class Volatility63d(PrimitiveFactor):
    FEATURE = features.Volatility63d

class Volatility126d(PrimitiveFactor):
    FEATURE = features.Volatility126d

class Volatility252d(PrimitiveFactor):
    FEATURE = features.Volatility252d
