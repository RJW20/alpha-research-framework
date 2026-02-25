from typing import Any, ClassVar, override

import alpha_research_framework.features as features
import alpha_research_framework.market_data as md
from alpha_research_framework.alphas.alpha import Alpha
from alpha_research_framework.universe import CrossSection
from alpha_research_framework.window import Window


class Volatility(Alpha, abstract=True):
    """
    Abstract base class for cross-sectional volatility alphas with automatic
    subclass validation.

    Any concrete subclass must define:
    - `ID`: `str` - unique identifier
    - `CATEGORY`: `str` - logical grouping label
    - `LOOKBACK`: `Window` - duration of rolling volatility to use in signal
    calculation
    - `HORIZONS`: `set[Window]` - prediction horizons for which the alpha will
    be evaluated against
    """

    LOOKBACK: ClassVar[Window]

    _VOLATILITY_LOOKBACK: ClassVar[type[features.Volatility]]

    def __init_subclass__(cls, abstract: bool = False, **kwargs: Any) -> None:
        """
        If `abstract=False` asserts definition and type of `LOOKBACK` and
        configures `DEPENDENCIES`.
        """

        if not abstract:
            cls.assert_class_var(name="LOOKBACK", type=Window)
            cls._VOLATILITY_LOOKBACK, = (
                Alpha._windows_to_volatilities(cls.LOOKBACK)
            )
            cls.DEPENDENCIES = {cls._VOLATILITY_LOOKBACK}
        
        kwargs["abstract"] = abstract
        super().__init_subclass__(**kwargs)

    @classmethod
    @override
    def compute(cls, x: CrossSection) -> md.Array:
        """`a_t = -sigma_t`"""

        return x[cls._VOLATILITY_LOOKBACK.ID] * -1
