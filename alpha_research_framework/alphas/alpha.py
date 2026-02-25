from abc import abstractmethod
from typing import Any, ClassVar

import alpha_research_framework.market_data as md
from alpha_research_framework.class_var_validator import ClassVarValidator
from alpha_research_framework.features import (
    DailyFutureReturns,
    DailyReturns,
    Feature,
    HalfYearlyFutureReturns,
    HalfYearlyReturns,
    MonthlyFutureReturns,
    MonthlyReturns,
    QuarterlyFutureReturns,
    QuarterlyReturns,
    WeeklyFutureReturns,
    WeeklyReturns,
    YearlyFutureReturns,
    YearlyReturns,
)
from alpha_research_framework.features.dependency_error import DependencyError
from alpha_research_framework.operator import Operator
from alpha_research_framework.universe import CrossSection
from alpha_research_framework.window import Window


class Alpha(Operator, ClassVarValidator, registry_root=True, abstract=True):
    """
    Abstract base class for cross-sectional alphas with automatic subclass
    validation and runtime missing dependency error reporting.

    Any concrete subclass must define:
    - `ID`: `str` - unique identifier
    - `CATEGORY`: `str` - logical grouping label
    - `DEPENDENCIES`: `set[Feature]` - prerequisite features this alpha needs to
    generate a signal
    - `HORIZONS`: `set[Window]` - prediction horizons for which the alpha will
    be evaluated against
    - `compute(cls, x: CrossSection) -> md.Array:` - classmethod for calculating
    the alpha
    """

    CATEGORY: ClassVar[str]
    DEPENDENCIES: ClassVar[set[type[Feature]]]
    HORIZONS: ClassVar[set[Window]]

    def __init_subclass__(cls, abstract: bool = False, **kwargs: Any) -> None:
        """
        If `abstract=False` asserts definition and type of `CATEGORY`,
        `DEPENDENCIES` and `HORIZONS` and wraps `compute` with error reporting.
        """

        kwargs["abstract"] = abstract
        super().__init_subclass__(**kwargs)

        if abstract:
            return
        
        cls.assert_class_var(name="CATEGORY", type=str, bad_values={""})
        cls.assert_class_var_container(
            name="DEPENDENCIES",
            container_type=set,
            element_type=type,
        )
        cls.assert_class_var_container(
            name="HORIZONS",
            container_type=set,
            element_type=Window,
        )

        cls.DEPENDENCIES.update(
            set(Alpha._windows_to_returns(*cls.HORIZONS, future=True))
        )
        cls._wrap_compute()

    @classmethod
    @abstractmethod
    def compute(cls, x: CrossSection) -> md.Array:
        """
        Return an `md.Array` containing raw cross-sectional alpha signal per
        stock.
        """
        ...

    @staticmethod
    def _windows_to_returns(
        *windows: Window,
        future: bool=False
    ) -> tuple[type[Feature],...]:
        """
        Return a `Returns` or `FutureReturns` feature pertaining to each window
        in `Windows`.
        """

        window_to_returns: dict[Window, type[Feature]] = dict()
        if not future:
            window_to_returns = {
                Window.DAY: DailyReturns,
                Window.WEEK: WeeklyReturns,
                Window.MONTH: MonthlyReturns,
                Window.QUARTER: QuarterlyReturns,
                Window.HALF_YEAR: HalfYearlyReturns,
                Window.YEAR: YearlyReturns,
            }
        else:
            window_to_returns = {
                Window.DAY: DailyFutureReturns,
                Window.WEEK: WeeklyFutureReturns,
                Window.MONTH: MonthlyFutureReturns,
                Window.QUARTER: QuarterlyFutureReturns,
                Window.HALF_YEAR: HalfYearlyFutureReturns,
                Window.YEAR: YearlyFutureReturns,
            }
        return tuple(window_to_returns[window] for window in windows)

    @classmethod
    def _wrap_compute(cls) -> None:
        """
        Wrap `compute` with error reporting for when `cls.DEPENDENCIES` or the
        `cross-section` argument do not contain all required dependencies.
        """

        original = getattr(cls, "compute")
        if getattr(original, "__isabstractmethod__", False):
            return

        def wrapper(cls: type[Alpha], x: CrossSection) -> md.Array:
            try:
                return original(x)
            except KeyError as e:
                missing_dependency = Feature.from_id(e.args[0])
                if missing_dependency not in cls.DEPENDENCIES:
                    raise DependencyError(
                        f"Alpha {cls.__name__} cannot be computed: to ensure "
                        f"{missing_dependency.__name__} is available please "
                        f"add it to {cls.__name__}.DEPENDENCIES"
                    )
                raise DependencyError(
                    f"Alpha {cls.__name__} cannot be computed: missing "
                    f"dependency {missing_dependency.__name__}"
                )

        setattr(cls, "compute", classmethod(wrapper))
