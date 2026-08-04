from typing import Any, ClassVar, override

import alpha_research_framework.market_data as md
import alpha_research_framework.observables as observables

from .cache import Cache
from .series import Series


class ObservableSeries(Series, abstract=True):
    """
    Abstract base class for market series whose values are read directly from
    the market data with automatic subclass validation.

    Any concrete subclass must define:
    - `TAG`: `Series.Tag` (`PREDICTOR` or `TARGET`) - usage classification
    - `OBSERVABLE`: `type[observables.Observable]`: observable extracted from
    the market data
    """

    OBSERVABLE: ClassVar[type[observables.Observable]]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Asserts definition and type of `OBSERVABLE`."""

        super().__init_subclass__(**kwargs)
        cls.assert_class_var_subtype(
            name="OBSERVABLE",
            base_type=observables.Observable,
        )

    @classmethod
    @override
    def compute(
        cls,
        market_data: md.MarketData,
        cache: Cache,
        allocator: md.Allocator,
    ) -> md.Array:
        """
        Return a reference to the `md.Array` pertaining to `OBSERVABLE` straight
        from `market_data`.

        Raises a `ValueError` if the required observable is not present in the
        market data.
        """
        
        try:
            return cls._compute(market_data)
        except KeyError as e:
            raise ValueError(
                f"Market series {cls.__name__} could not be computed: "
                "market data missing required observable "
                f"{cls.OBSERVABLE.__name__}"
            ) from e

    @classmethod
    def _compute(cls, market_data: md.MarketData) -> md.Array:
        return market_data[cls.OBSERVABLE]
