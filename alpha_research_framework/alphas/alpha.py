from typing import Any, ClassVar, override

import alpha_research_framework.cross_section as xs
import alpha_research_framework.features as features
from alpha_research_framework.class_var_validator import ClassVarValidator
from alpha_research_framework.registrable import Registrable
from alpha_research_framework.window import Window

from . import factors


class Alpha(
    factors.Factor,
    Registrable,
    ClassVarValidator,
    registry_root=True,
    abstract=True,
):
    """
    Abstract base class for cross-sectional alphas with automatic subclass
    validation and runtime missing dependency error reporting.

    Any concrete subclass must define:
    - `ID`: `str` - unique identifier
    - `CATEGORY`: `str` - logical grouping label
    - `SIGNAL`: `type[factors.Factor]` - factor to use as cross-sectional signal
    - `HORIZONS`: `set[Window]` - prediction horizons for which the alpha will
    be evaluated against
    """

    CATEGORY: ClassVar[str]
    SIGNAL: ClassVar[type[factors.Factor]]
    HORIZONS: ClassVar[set[Window]]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """
        Asserts definition and type of `CATEGORY`, `SIGNAL` and `HORIZONS`,
        and configures `REQUIRED_FEATURES` as the union of
        `SIGNAL.REQUIRED_FEATURES` and a set of `ForwardReturns` features for
        all `HORIZONS`.
        """
        
        cls.assert_class_var("CATEGORY", type=str, bad_values={""})
        cls.assert_class_var_subtype("SIGNAL", base_type=factors.Factor)
        cls.assert_class_var_container(
            name="HORIZONS",
            container_type=set,
            element_type=Window,
        )

        cls.REQUIRED_FEATURES = (                                               # type: ignore
            cls.SIGNAL.REQUIRED_FEATURES |
            set(Alpha._windows_to_forward_returns(*cls.HORIZONS))
        )

        super().__init_subclass__(**kwargs)

    @classmethod
    @override
    def compute(
        cls,
        x: xs.CrossSection,
        cache: factors.FactorCache,
    ) -> xs.Array:
        return cls.SIGNAL.compute(x, cache)

    @staticmethod
    def _windows_to_forward_returns(
        *windows: Window,
    ) -> tuple[type[features.Feature],...]:
        """
        Return a `ForwardReturns` feature pertaining to each window in
        `Windows`.
        """

        window_to_forward_returns = {
            Window.DAY:         features.ForwardReturns1d,
            Window.WEEK:        features.ForwardReturns5d,
            Window.MONTH:       features.ForwardReturns20d,
            Window.QUARTER:     features.ForwardReturns63d,
            Window.HALF_YEAR:   features.ForwardReturns126d,
            Window.YEAR:        features.ForwardReturns252d,
        }
        return tuple(window_to_forward_returns[window] for window in windows)
