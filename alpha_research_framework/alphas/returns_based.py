from typing import Any, ClassVar, override

import alpha_research_framework.features as features
import alpha_research_framework.market_data as md
from alpha_research_framework.alphas.alpha import Alpha
from alpha_research_framework.universe import CrossSection
from alpha_research_framework.window import Window


class ReturnsBased(Alpha, abstract=True):
    """
    Abstract base class for returns-based cross-sectional alphas with automatic
    subclass validation.

    Any concrete subclass must define:
    - `ID`: `str` - unique identifier
    - `CATEGORY`: `str` - logical grouping label
    - `LOOKBACK`: `Window` - duration of past returns to use in signal
    calculation
    - (optional) `SKIP`: `Window` - duration of past returns to ignore in signal
    calculation (must be less than `LOOKBACK`)
    - `HORIZONS`: `set[Window]` - prediction horizons for which the alpha will
    be evaluated against
    """

    LOOKBACK: ClassVar[Window]
    SKIP: ClassVar[Window]

    _RETURNS_LOOKBACK: ClassVar[type[features.Returns]]
    _RETURNS_SKIP: ClassVar[type[features.Returns] | None]

    def __init_subclass__(cls, abstract: bool = False, **kwargs: Any) -> None:
        """
        If `abstract=False` asserts definition and type of `LOOKBACK`,
        definition, type and value (compared to `LOOKBACK`) of `SKIP` and
        configures `DEPENDENCIES`.
        """

        if not abstract:

            cls.assert_class_var(name="LOOKBACK", type=Window)

            try:
                cls.assert_class_var(name="SKIP", type=Window)
                if not cls.SKIP < cls.LOOKBACK:
                    raise ValueError(
                        f"{cls.__name__}.SKIP must be less than "
                        f"{cls.__name__}.LOOKBACK"
                    )
                cls._RETURNS_LOOKBACK, cls._RETURNS_SKIP = (
                    Alpha._windows_to_returns(cls.LOOKBACK, cls.SKIP)
                )
                cls.DEPENDENCIES = {cls._RETURNS_LOOKBACK, cls._RETURNS_SKIP}
            except AttributeError:
                cls._RETURNS_LOOKBACK, cls._RETURNS_SKIP = (
                    *Alpha._windows_to_returns(cls.LOOKBACK),
                    None,
                )
                cls.DEPENDENCIES = {cls._RETURNS_LOOKBACK}

        kwargs["abstract"] = abstract
        super().__init_subclass__(**kwargs)

    @classmethod
    @override
    def compute(cls, x: CrossSection) -> md.Array:
        """`a_t = r_{t-lookback} - r_{t-skip}`"""

        if cls._RETURNS_SKIP is not None:
            return x[cls._RETURNS_LOOKBACK.ID] - x[cls._RETURNS_SKIP.ID]
        else:
            return x[cls._RETURNS_LOOKBACK.ID]
