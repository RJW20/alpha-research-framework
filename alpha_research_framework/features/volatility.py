from typing import Any, ClassVar, override

import alpha_research_framework.market_data as md
from alpha_research_framework.class_var_validator import ClassVarValidator
from alpha_research_framework.features.feature import Feature
from alpha_research_framework.features.features import Features
from alpha_research_framework.features.returns import DailyReturns
from alpha_research_framework.window import Window


class Volatility(Feature, ClassVarValidator, abstract=True):
    """
    Abstract base class for volatility features with automatic subclass
    validation.

    Any concrete subclass must define:
    - `LOOKBACK`: `Window` - duration of period into the past to track rolling
    volatility over
    """

    TAG = Feature.Tag.PREDICTOR
    DEPENDENCIES = {DailyReturns}
    LOOKBACK: ClassVar[Window]

    def __init_subclass__(cls, abstract: bool = False, **kwargs: Any) -> None:
        """
        If `abstract=False` asserts definition and type of `LOOKBACK` and sets
        `ID`.
        """

        if not abstract:
            cls.assert_class_var(name="LOOKBACK", type=Window)
            cls.ID = f"vol_{cls.LOOKBACK.value}d"

        kwargs["abstract"] = abstract
        super().__init_subclass__(**kwargs)
    
    @classmethod
    @override
    def compute(
        cls,
        market_data: md.MarketData,
        features: Features,
        out: md.Array
    ) -> None:
        """
        `sigma_t = sqrt(sum(r_{t-lookback+1}, ..., r_{t}) / (lookback - 1))`
        """

        lookback = cls.LOOKBACK.value
        ret_1d = features[DailyReturns.ID]
        cls._rolling_std(ret_1d, lookback, out)


class DailyVolatility(Volatility):
    LOOKBACK = Window.DAY

class WeeklyVolatility(Volatility):
    LOOKBACK = Window.WEEK

class MonthlyVolatility(Volatility):
    LOOKBACK = Window.MONTH

class QuarterlyVolatility(Volatility):
    LOOKBACK = Window.QUARTER

class HalfYearlyVolatility(Volatility):
    LOOKBACK = Window.HALF_YEAR

class YearlyVolatility(Volatility):
    LOOKBACK = Window.YEAR
