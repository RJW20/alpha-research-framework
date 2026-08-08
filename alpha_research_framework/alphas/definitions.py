import alpha_research_framework.signals as signals
from alpha_research_framework.window import Window

from .alpha import Alpha

# ---------------------------------- Notes -------------------------------------
# Contained here is a full definition of all built-in Alphas that this library
# provides.
# For information on creating custom Alphas see the "Alphas" section of the
# documentation.
# ---------------------------------- Notes -------------------------------------

# ------------------------------------------------------------------------------
# Short Term Reversal
# Hypothesis: large short-term moves are often an overreaction
# ------------------------------------------------------------------------------

class Reversal1d(Alpha):
    """
    - Signal: `-returns_1d`
    - Horizons: `1d`, `5d`
    """
    ID = "reversal_1d"
    CATEGORY = "short_term_reversal"
    SIGNAL = -signals.Returns1d
    HORIZONS = {Window.DAY, Window.WEEK}

class Reversal5d(Alpha):
    """
    - Signal: `-returns_5d`
    - Horizons: `1d`, `5d`
    """
    ID = "reversal_5d"
    CATEGORY = "short_term_reversal"
    SIGNAL = -signals.Returns5d
    HORIZONS = {Window.DAY, Window.WEEK}
    
# ------------------------------------------------------------------------------
# Momentum
# Hypothesis: price trends persist
# ------------------------------------------------------------------------------

class Momentum20d(Alpha):
    """
    - Signal: `returns_20d`
    - Horizons: `1d`, `5d`, `20d`
    """
    ID = "momentum_20d"
    CATEGORY = "momentum"
    SIGNAL = signals.Returns20d
    HORIZONS = {Window.DAY, Window.WEEK, Window.MONTH}

class Momentum12m(Alpha):
    """
    - Signal: `returns_252d`
    - Horizons: `20d`, `63d`, `126d`, `252d`
    """
    ID = "momentum_12m"
    CATEGORY = "momentum"
    SIGNAL = signals.Returns252d
    HORIZONS = {Window.MONTH, Window.QUARTER, Window.HALF_YEAR, Window.YEAR}

class Momentum12m1m(Alpha):
    """
    - Signal: `returns_252d - returns_20d`
    - Horizons: `20d`, `63d`, `126d`, `252d`
    """
    ID = "momentum_12m_1m"
    CATEGORY = "momentum"
    SIGNAL = signals.Returns252d - signals.Returns20d
    SKIP = Window.MONTH
    HORIZONS = {Window.MONTH, Window.QUARTER, Window.HALF_YEAR, Window.YEAR}

class RiskAdjustedMomentum12m1m(Alpha):
    """
    - Signal: `(returns_252d - returns_20d) / volatility_252d`
    - Horizons: `20d`, `63d`, `126d`, `252d`
    """
    ID = "risk_adjusted_momentum_12m_1m"
    CATEGORY = "momentum"
    SIGNAL = (signals.Returns252d - signals.Returns20d) / signals.Volatility252d
    HORIZONS = {Window.MONTH, Window.QUARTER, Window.HALF_YEAR, Window.YEAR}

# ------------------------------------------------------------------------------
# Volatility
# Hypothesis: market overprices volatility
# ------------------------------------------------------------------------------

class Volatility20d(Alpha):
    """
    - Signal = `-volatility_20d`
    - Horizons = `1d`, `5d`, `20d`
    """
    ID = "volatility_20d"
    CATEGORY = "volatility"
    SIGNAL = -signals.Volatility20d
    HORIZONS = {Window.DAY, Window.WEEK, Window.MONTH}

class Volatility12m(Alpha):
    """
    - Signal = `-volatility_252d`
    - Horizons = `20d`, `63d`, `126d`, `252d`
    """
    ID = "volatility_12m"
    CATEGORY = "volatility"
    SIGNAL = -signals.Volatility252d
    HORIZONS = {Window.MONTH, Window.QUARTER, Window.HALF_YEAR, Window.YEAR}
