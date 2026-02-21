import alpha_research_framework.features as features
import alpha_research_framework.market_data as md
from alpha_research_framework.alphas.alpha import Alpha
from alpha_research_framework.alphas.alpha_error import AlphaError
from alpha_research_framework.features import FeatureSpec
from alpha_research_framework.universe import CrossSection
from alpha_research_framework.window import Window


class Volatility(Alpha):
    """
    Abstract base class for cross-sectional volatility alphas with automatic
    subclass validation.

    Any concrete subclass must define:
    - NAME: str - unique identifier
    - CATEGORY: str - logical grouping label
    - LOOKBACK: Window - period into the past to track rolling volatility over
    - HORIZONS: set[Window] - prediction horizons for which the alpha will be
    evaluated
    """

    __abstract__ = True

    LOOKBACK: Window | None = None

    def __init_subclass__(cls) -> None:
        """Validate definition and type of LOOKBACK."""

        super().__init_subclass__()

        if cls is Volatility:
            return
        
        if cls.__dict__.get("__abstract__", False):
            return

        if cls.LOOKBACK is None:
            raise AlphaError(f"{cls.__name__} must define LOOKBACK.")
        if not isinstance(cls.LOOKBACK, Window):
            raise TypeError(f"{cls.__name__}.LOOKBACK must be of type Window.")

    def compute(self, x: CrossSection) -> md.Array:
        """a_t = -s_t"""

        return x[self._volatility.name] * -1
    
    def _init_dependencies(self) -> set[FeatureSpec]:
        """Create a dependency for volatility over LOOKBACK."""
        
        self._volatility = FeatureSpec(features.Volatility, self.LOOKBACK)
        return {self._volatility}
