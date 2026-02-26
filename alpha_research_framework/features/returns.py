from typing import Any, ClassVar, override

import numpy as np

import alpha_research_framework.market_data as md
from alpha_research_framework.class_var_validator import ClassVarValidator
from alpha_research_framework.features.feature import Feature
from alpha_research_framework.features.features import Features
from alpha_research_framework.features.log_price import LogPrice
from alpha_research_framework.window import Window


class Returns(Feature, ClassVarValidator, abstract=True):
    """
    Abstract base class for returns features with automatic subclass validation.

    Any concrete subclass must define:
    - `LOOKBACK`: `Window` - duration of period into the past to track returns
    over
    """

    TAG = Feature.Tag.PREDICTOR
    DEPENDENCIES = {LogPrice}
    LOOKBACK: ClassVar[Window]

    def __init_subclass__(cls, abstract: bool=False, **kwargs: Any) -> None:
        """
        If `abstract=False` asserts definition and type of `LOOKBACK` and sets
        `ID`.
        """

        if not abstract:
            cls.assert_class_var(name="LOOKBACK", type=Window)
            cls.ID = f"returns_{cls.LOOKBACK.value}d"
            
        kwargs["abstract"] = abstract
        super().__init_subclass__(**kwargs)

    @classmethod
    @override
    def compute(
        cls,
        market_data: md.MarketData,
        features: Features,
        out: md.Array,
    ) -> None:
        """`r_t = log(p_t) - log(p_{t-lookback})`"""

        lookback = cls.LOOKBACK.value
        log_price = features[LogPrice]
        out[:lookback] = np.nan
        out[lookback:] = log_price[lookback:] - log_price[:-lookback]


class DailyReturns(Returns):
    LOOKBACK = Window.DAY

class WeeklyReturns(Returns):
    LOOKBACK = Window.WEEK

class MonthlyReturns(Returns):
    LOOKBACK = Window.MONTH

class QuarterlyReturns(Returns):
    LOOKBACK = Window.QUARTER

class HalfYearlyReturns(Returns):
    LOOKBACK = Window.HALF_YEAR

class YearlyReturns(Returns):
    LOOKBACK = Window.YEAR
