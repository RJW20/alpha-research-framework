from typing import Any, ClassVar, override

import alpha_research_framework.cross_section as xs
import alpha_research_framework.signals as signals
from alpha_research_framework.class_var_validator import ClassVarValidator
from alpha_research_framework.operator import Operator
from alpha_research_framework.registrable import Registrable
from alpha_research_framework.window import Window


class Alpha(
    Operator,
    Registrable,
    ClassVarValidator,
    registry_root=True,
):
    """
    Abstract base class for cross-sectional alphas with automatic subclass
    validation and runtime missing dependency error reporting.

    Any concrete subclass must define:
    - `ID`: `str` - unique identifier
    - `CATEGORY`: `str` - logical grouping label
    - `SIGNAL`: `type[signals.Signal]` - cross-sectional signal
    - `HORIZONS`: `set[Window]` - prediction horizons for which the alpha will
    be evaluated against
    """

    CATEGORY: ClassVar[str]
    SIGNAL: ClassVar[type[signals.Signal]]
    HORIZONS: ClassVar[set[Window]]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """
        Asserts definition and type of `CATEGORY`, `SIGNAL` and `HORIZONS`.
        """

        super().__init_subclass__(**kwargs)
        cls.assert_class_var("CATEGORY", type=str, bad_values={""})
        cls.assert_class_var_subtype("SIGNAL", base_type=signals.Signal)
        cls.assert_class_var_container(
            name="HORIZONS",
            container_type=set,
            element_type=Window,
        )

    @classmethod
    @override
    def compute(
        cls,
        cross_section: xs.CrossSection,
        cache: signals.Cache,
    ) -> xs.Array:
        return cls.SIGNAL.compute(cross_section, cache)
