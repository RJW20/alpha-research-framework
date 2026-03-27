from dataclasses import dataclass

import numpy as np
import pandas as pd

import alpha_research_framework.cross_section as xs
import alpha_research_framework.features as features
import alpha_research_framework.market_data as md

from .calendar import Calendar


@dataclass(frozen=True)
class Universe:
    """
    Cross-sectional read-only universe backed by memory-mapped arrays.

    Holds a global calendar, time * stock market data (e.g. prices, volumes)
    and calculated features, along with a time-varying in-universe mask.
    Construction should be done via `build_universe`.

    Supports indexing via `[pd.Timestamp]`.

    Attributes
    ----------
    shape : tuple[int,int]
        `(T, N)` where `T` is the number of trading days with data and `N` is
        number of stocks with data.
    dates : pd.Index
        Listing of all dates with market data.
    """

    shape: tuple[int, int]
    _calendar: Calendar
    _mask: np.memmap
    _features: dict[type[features.Feature], md.Array]

    @property
    def dates(self) -> pd.Index:
        """Return all dates with data."""
        return self._calendar.index
    
    def __getitem__(
        self,
        key: pd.Timestamp,
    ) -> tuple[xs.CrossSection, xs.CrossSection]:
        t = self._calendar.t(key)
        """
        Return a tuple of `xs.CrossSection`s, containing `PREDICTOR`, `TARGET`
        features for all stocks in-universe at the given timestamp.
        """

        return (
            self._cross_section(t, features.Feature.Tag.PREDICTOR),
            self._cross_section(t, features.Feature.Tag.TARGET),
        )
    
    def _cross_section(
        self,
        t: int,
        tag: features.Feature.Tag,
    ) -> xs.CrossSection:
        """
        Return an `xs.CrossSection` containing all features with given `tag` for
        all stocks in-universe at time `t`.
        """

        mask = self._mask[t, :]
        return xs.CrossSection(
            {
                feature: values[t, mask]
                for feature, values in self._features.items()
                if feature.TAG == tag
            }
        )
