from typing import Any, ClassVar, override

import alpha_research_framework.features as features
import alpha_research_framework.market_data as md
from alpha_research_framework.alphas.alpha import Alpha
from alpha_research_framework.universe import CrossSection
from alpha_research_framework.window import Window


class RiskAdjustedReturns(Alpha, abstract=True):
    """
    Abstract base class for cross-sectional risk-adjusted returns alphas with
    automatic subclass validation.

    Any concrete subclass must define:
    - `ID`: `str` - unique identifier
    - `CATEGORY`: `str` - logical grouping label
    - `RETURNS_LOOKBACK`: `Window` - duration of past returns to use in signal
    calculation
    - (optional) `RETURNS_SKIP`: `Window` - duration of past returns to ignore
    in signal calculation (must be less than `LOOKBACK`)
    - `VOLATILITY_LOOKBACK`: `Window` - duration of rolling volatility to use in
    signal calculation
    - `HORIZONS`: `set[Window]` - prediction horizons for which the alpha will
    be evaluated against
    """

    RETURNS_LOOKBACK: ClassVar[Window]
    RETURNS_SKIP: ClassVar[Window]
    VOLATILITY_LOOKBACK: ClassVar[Window]

    _RETURNS_LOOKBACK: ClassVar[type[features.Returns]]
    _RETURNS_SKIP: ClassVar[type[features.Returns] | None]
    _VOLATILITY_LOOKBACK: ClassVar[type[features.Volatility]]

    def __init_subclass__(cls, abstract: bool = False, **kwargs: Any) -> None:
        """
        If `abstract=False` asserts definition and type of `RETURNS_LOOKBACK`
        and `VOLATILITY_LOOKBACK`, type and value (compared to
        `RETURNS_LOOKBACK`) of `RETURNS_SKIP` if defined and configures
        `DEPENDENCIES`.
        """

        if not abstract:

            cls.assert_class_var(name="RETURNS_LOOKBACK", type=Window)
            cls._RETURNS_LOOKBACK, = Alpha._windows_to_returns(
                cls.RETURNS_LOOKBACK,
            )
            cls.DEPENDENCIES = {cls._RETURNS_LOOKBACK}

            try:
                cls.assert_class_var(name="RETURNS_SKIP", type=Window)
                if not cls.RETURNS_SKIP < cls.RETURNS_LOOKBACK:
                    raise ValueError(
                        f"{cls.__name__}.RETURNS_SKIP must be less than "
                        f"{cls.__name__}.RETURNS_LOOKBACK"
                    )
                cls._RETURNS_SKIP, = Alpha._windows_to_returns(cls.RETURNS_SKIP)
                cls.DEPENDENCIES.add(cls._RETURNS_SKIP)

            except AttributeError:
                cls._RETURNS_SKIP = None

            cls.assert_class_var(name="VOLATILITY_LOOKBACK", type=Window)
            cls._VOLATILITY_LOOKBACK, = (
                Alpha._windows_to_volatilities(cls.VOLATILITY_LOOKBACK)
            )
            cls.DEPENDENCIES.add(cls._VOLATILITY_LOOKBACK)

        kwargs["abstract"] = abstract
        super().__init_subclass__(**kwargs)

    @classmethod
    @override
    def compute(cls, x: CrossSection) -> md.Array:
        """`a_t = (r_{t-returns_lookback} - r_{t-returns_skip}) / sigma_t`"""

        if cls._RETURNS_SKIP is not None:
            return (
                (x[cls._RETURNS_LOOKBACK.ID] - x[cls._RETURNS_SKIP.ID]) /
                x[cls._VOLATILITY_LOOKBACK.ID]
            )
        else:
            return x[cls._RETURNS_LOOKBACK.ID] / x[cls._VOLATILITY_LOOKBACK.ID]
