from typing import Any, ClassVar, override

import numpy as np

import alpha_research_framework.market_data as md
from alpha_research_framework.class_var_validator import ClassVarValidator
from alpha_research_framework.features.feature import Feature
from alpha_research_framework.features.features import Features
from alpha_research_framework.features.log_price import LogPrice
from alpha_research_framework.window import Window


class FutureReturns(Feature, ClassVarValidator, abstract=True):
    """
    Abstract base class for future returns features with automatic subclass
    validation.

    Any concrete subclass must define:
    - `HORIZON`: `Window` - duration of period into the future to track returns
    over
    """

    TAG = Feature.Tag.TARGET
    DEPENDENCIES = {LogPrice}
    HORIZON: ClassVar[Window]

    def __init_subclass__(cls, abstract: bool = False, **kwargs: Any) -> None:
        """
        If `abstract=False` asserts definition and type of `HORIZON` and sets
        `ID`.
        """

        if not abstract:
            cls.assert_class_var(name="HORIZON", type=Window)
            cls.ID = f"future_returns_{cls.HORIZON.value}d"

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
        """`r_t = log(p_{t+horizon}) - log(p_t)`"""

        horizon = cls.HORIZON.value
        log_price = features[LogPrice.ID]
        out[:-horizon] = log_price[horizon:] - log_price[:-horizon]
        out[-horizon:] = np.nan


class DailyFutureReturns(FutureReturns):
    HORIZON = Window.DAY

class WeeklyFutureReturns(FutureReturns):
    HORIZON = Window.WEEK

class MonthlyFutureReturns(FutureReturns):
    HORIZON = Window.MONTH

class QuarterlyFutureReturns(FutureReturns):
    HORIZON = Window.QUARTER

class HalfYearlyFutureReturns(FutureReturns):
    HORIZON = Window.HALF_YEAR

class YearlyFutureReturns(FutureReturns):
    HORIZON = Window.YEAR
