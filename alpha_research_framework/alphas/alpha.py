from abc import ABC, abstractmethod

import alpha_research_framework.market_data as md
from alpha_research_framework.alphas.alpha_error import AlphaError
from alpha_research_framework.dependent import Dependent
from alpha_research_framework.features import FeatureSpec, FutureReturns
from alpha_research_framework.universe import CrossSection
from alpha_research_framework.window import Window


class Alpha(Dependent[FeatureSpec], ABC):
    """
    Abstract base class for cross-sectional alphas with automatic subclass
    validation, runtime dependency enforcement and registry management.

    Any concrete subclass must define:
    - NAME: str - unique identifier
    - CATEGORY: str - logical grouping label
    - HORIZONS: set[Window] - prediction horizons for which the alpha will be
    evaluated
    - _init_dependencies(self) -> set[FeatureSpec] - features the alpha will
    need to generate a signal
    - compute(self, x: CrossSection) -> md.Array - signal per stock in
    cross-section
    Concrete compute() methods are automatically wrapped to enforce runtime
    feature dependency checks.
    """

    __registry__: dict[str, type["Alpha"]] = dict()

    __abstract__ = True
    __dependency_type__ = FeatureSpec

    NAME: str | None = None
    CATEGORY: str | None = None

    HORIZONS: set[Window] | None = None

    def __init_subclass__(cls) -> None:
        """
        Validate definition, type and value of NAME and CATEGORY and definition
        and type of HORIZONS, wrap compute with runtime dependency check and add
        subclass to registry.
        """

        super().__init_subclass__()

        if cls is Alpha:
            return
        
        if cls.__dict__.get("__abstract__", False):
            return

        if cls.NAME is None:
            raise AlphaError(f"{cls.__name__} must define NAME.")
        if not isinstance(cls.NAME, str):
            raise TypeError(f"{cls.__name__}.NAME must be of type str.")
        if not cls.NAME:
            raise ValueError(f"{cls.__name__}.NAME cannot be empty.")
        if cls.NAME in Alpha.__registry__:
            raise AlphaError(
                f"{cls.__name__}.NAME must be unique (alpha with NAME "
                f"'{cls.NAME}' already exists)."
            )
        
        if cls.CATEGORY is None:
            raise AlphaError(f"{cls.__name__} must define CATEGORY.")
        if not isinstance(cls.CATEGORY, str):
            raise TypeError(f"{cls.__name__}.CATEGORY must be of type str.")
        if not cls.CATEGORY:
            raise ValueError(f"{cls.__name__}.CATEGORY cannot be empty.")
        
        if cls.HORIZONS is None:
            raise AlphaError(f"{cls.__name__} must define HORIZONS.")
        if (
            not isinstance(cls.HORIZONS, set) or
            not all(isinstance(h, Window) for h in cls.HORIZONS)
        ):
            raise TypeError(
                f"{cls.__name__}.HORIZONS must be of type set[Window]."
            )

        cls._wrap_compute()

        Alpha.__registry__[cls.NAME] = cls

    @staticmethod
    def from_name(name: str) -> type["Alpha"]:
        """Return the alpha with NAME = name."""
        if name not in Alpha.__registry__:
            raise AlphaError(f"Alpha with NAME '{name}' does not exist.")
        return Alpha.__registry__[name]
    
    @property
    def required_features(self) -> set[FeatureSpec]:
        """
        Return a set containing feature specs of all dependencies and future
        returns.
        """
        return set(
            self.dependencies |
            set(FeatureSpec(FutureReturns, h) for h in self.HORIZONS)
        )

    @abstractmethod
    def compute(self, x: CrossSection) -> md.Array:
        """
        Return an md.Array containing raw cross-sectional alpha signal per
        stock.
        """
        ...

    @classmethod
    def _wrap_compute(cls) -> None:
        """Wrap compute() to enforce runtime dependency validation."""

        original = getattr(cls, "compute", None)
        if original is None or getattr(original, "__isabstractmethod__", False):
            return

        def wrapper(self: Alpha, x: CrossSection) -> md.Array:
            dependencies = {feature.name for feature in self._dependencies}
            missing = dependencies - x.keys()
            if missing:
                raise AlphaError(
                    f"Alpha {self.NAME} cannot be computed: missing feature "
                    f"dependencies {missing}."
                )
            return original(self, x)

        setattr(cls, "compute", wrapper)
