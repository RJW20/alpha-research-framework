import alpha_research_framework.market_data as md
from alpha_research_framework.alphas.returns_based import ReturnsBased
from alpha_research_framework.alphas.volatility import Volatility
from alpha_research_framework.universe import CrossSection
from alpha_research_framework.window import Window


class Reversal1d(ReturnsBased):
    """
    Hypothesis: large short-term moves are often an overreaction
    Signal = -returns_1d
    Horizons = 1d, 5d
    """

    NAME = "reversal_1d"
    CATEGORY = "short_term_reversal"
    LOOKBACK = Window.DAY
    HORIZONS = {Window.DAY, Window.WEEK}

    def compute(self, x: CrossSection) -> md.Array:
        return super().compute(x) * -1
    

class Reversal5d(ReturnsBased):
    """
    Hypothesis: large short-term moves are often an overreaction
    Signal = -returns_5d
    Horizons = 1d, 5d
    """

    NAME = "reversal_5d"
    CATEGORY = "short_term_reversal"
    LOOKBACK = Window.WEEK
    HORIZONS = {Window.DAY, Window.WEEK}

    def compute(self, x: CrossSection) -> md.Array:
        return super().compute(x) * -1


class Momentum12To1(ReturnsBased):
    """
    Hypothesis: price trends persist
    Signal = returns_252d - returns_20d
    Horizons = 20d, 63d, 126d, 252d
    """

    NAME = "momentum_12_1"
    CATEGORY = "momentum/trend"

    LOOKBACK = Window.YEAR
    SKIP = Window.MONTH
    HORIZONS = {Window.MONTH, Window.QUARTER, Window.HALF_YEAR, Window.YEAR}

        
class Volatility20d(Volatility):
    """
    Hypothesis: market overprices volatility
    Signal = -volatility_20d
    Horizons = 5d, 20d
    """

    NAME = "volatility_20d"
    CATEGORY = "volatility"

    LOOKBACK = Window.MONTH
    HORIZONS = {Window.WEEK, Window.MONTH}
