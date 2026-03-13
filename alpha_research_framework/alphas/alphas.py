from typing import override

import alpha_research_framework.market_data as md
from alpha_research_framework.alphas.returns_based import ReturnsBased
from alpha_research_framework.alphas.risk_adjusted_returns import (
    RiskAdjustedReturns,
)
from alpha_research_framework.alphas.volatility import Volatility
from alpha_research_framework.universe import CrossSection
from alpha_research_framework.window import Window

# -----------------------------------------------------------------------------
# Short Term Reversal
# Hypothesis: large short-term moves are often an overreaction
# -----------------------------------------------------------------------------

class Reversal1d(ReturnsBased):
    """
    - Signal: `-returns_1d`
    - Horizons: `1d`, `5d`
    """
    ID = "reversal_1d"
    CATEGORY = "short_term_reversal"
    LOOKBACK = Window.DAY
    HORIZONS = {Window.DAY, Window.WEEK}
    @classmethod
    @override
    def compute(cls, x: CrossSection) -> md.Array:
        return super().compute(x) * -1

class Reversal5d(ReturnsBased):
    """
    - Signal: `-returns_5d`
    - Horizons: `1d`, `5d`
    """
    ID = "reversal_5d"
    CATEGORY = "short_term_reversal"
    LOOKBACK = Window.WEEK
    HORIZONS = {Window.DAY, Window.WEEK}
    @classmethod
    @override
    def compute(cls, x: CrossSection) -> md.Array:
        return super().compute(x) * -1
    
# -----------------------------------------------------------------------------
# Momentum
# Hypothesis: price trends persist
# -----------------------------------------------------------------------------

class Momentum12m(ReturnsBased):
    """
    - Signal: `returns_252d`
    - Horizons: `20d`, `63d`, `126d`, `252d`
    """
    ID = "momentum_12m"
    CATEGORY = "momentum"
    LOOKBACK = Window.YEAR
    SKIP = Window.MONTH
    HORIZONS = {Window.MONTH, Window.QUARTER, Window.HALF_YEAR, Window.YEAR}

class Momentum12m1m(ReturnsBased):
    """
    - Signal: `returns_252d - returns_20d`
    - Horizons: `20d`, `63d`, `126d`, `252d`
    """
    ID = "momentum_12m_1m"
    CATEGORY = "momentum"
    LOOKBACK = Window.YEAR
    SKIP = Window.MONTH
    HORIZONS = {Window.MONTH, Window.QUARTER, Window.HALF_YEAR, Window.YEAR}

class RiskAdjustedMomentum12m1m(RiskAdjustedReturns):
    """
    - Signal: `(returns_252d - returns_20d) / volatility_252d`
    - Horizons: `20d`, `63d`, `126d`, `252d`
    """
    ID = "risk_adjusted_momentum_12m_1m"
    CATEGORY = "momentum"
    RETURNS_LOOKBACK = Window.YEAR
    RETURNS_SKIP = Window.MONTH
    VOLATILITY_LOOKBACK = Window.YEAR
    HORIZONS = {Window.MONTH, Window.QUARTER, Window.HALF_YEAR, Window.YEAR}

# -----------------------------------------------------------------------------
# Volatility
# Hypothesis: market overprices volatility
# -----------------------------------------------------------------------------

class Volatility20d(Volatility):
    """
    - Signal = `-volatility_20d`
    - Horizons = `1d`, `5d`, `20d`
    """
    ID = "volatility_20d"
    CATEGORY = "volatility"
    LOOKBACK = Window.MONTH
    HORIZONS = {Window.DAY, Window.WEEK, Window.MONTH}

class Volatility12m(Volatility):
    """
    - Signal = `-volatility_252d`
    - Horizons = `20d`, `63d`, `126d`, `252d`
    """
    ID = "volatility_12m"
    CATEGORY = "volatility"
    LOOKBACK = Window.YEAR
    HORIZONS = {Window.MONTH, Window.QUARTER, Window.HALF_YEAR, Window.YEAR}
