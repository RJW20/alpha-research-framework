import numpy as np
import numpy.typing as npt

import alpha_research_framework.features as features
from alpha_research_framework.alphas.alpha import Alpha
from alpha_research_framework.alphas.alpha_error import AlphaError
from alpha_research_framework.features import FeatureSpec
from alpha_research_framework.universe import CrossSection
from alpha_research_framework.window import Window


class ReturnsBased(Alpha):
    """
    Abstract base class for returns-based cross-sectional alphas with automatic
    subclass validation.

    Any concrete subclass must define:
    - NAME: str - unique identifier
    - CATEGORY: str - logical grouping label
    - LOOKBACK: Window - period into the past to start tracking returns
    - (optional) SKIP: Window - period into the past to stop tracking returns (
    must be less than LOOKBACK)
    - HORIZONS: set[Window] - prediction horizons for which the alpha will be
    evaluated
    """

    __abstract__ = True

    LOOKBACK: Window | None = None
    SKIP: Window | None = None

    def __init_subclass__(cls) -> None:
        """Validate definition, type and value of LOOKBACK and SKIP."""

        super().__init_subclass__()

        if cls is ReturnsBased:
            return
        
        if cls.__dict__.get("__abstract__", False):
            return

        if cls.LOOKBACK is None:
            raise AlphaError(f"{cls.__name__} must define LOOKBACK.")
        if not isinstance(cls.LOOKBACK, Window):
            raise TypeError(f"{cls.__name__}.LOOKBACK must be of type Window.")
    
        if cls.SKIP is not None:
            if not isinstance(cls.SKIP, Window):
                raise TypeError(f"{cls.__name__}.SKIP must be of type Window.")
            if not cls.SKIP < cls.LOOKBACK:
                raise ValueError(
                    f"{cls.__name__}.SKIP must be less than "
                    f"{cls.__name__}.LOOKBACK."
                )

    def compute(self, x: CrossSection) -> npt.NDArray[np.float32]:
        """a_t = r_{t-lookback} - r_{t-skip}"""

        if self._returns_skip is not None:
            return x[self._returns_lookback.name] - x[self._returns_skip.name]
        else:
            return x[self._returns_lookback.name]
        
    def _init_dependencies(self) -> set[FeatureSpec]:
        """
        Create a dependency for returns over LOOKBACK (and SKIP if present).
        """

        self._returns_lookback = FeatureSpec(features.Returns, self.LOOKBACK)
        dependencies = {self._returns_lookback}
        if self.SKIP is not None:
            self._returns_skip = FeatureSpec(features.Returns, self.SKIP)
            dependencies.add(self._returns_skip)
        else:
            self._returns_skip = None
        return dependencies
