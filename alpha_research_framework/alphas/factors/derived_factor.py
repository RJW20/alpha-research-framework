from abc import abstractmethod
from typing import override

import alpha_research_framework.cross_section as xs

from .factor import Factor
from .factor_cache import FactorCache


class DerivedFactor(Factor, abstract=True):
    """
    Abstract base class for factors derived from combinations of other factors.
    
    Should not be directly subclassed - derived factors may be created by
    combining `Factor`s via the operators `+`, `-`, `*`, `/`.
    """

    @classmethod
    @override
    def compute(cls, x: xs.CrossSection, cache: FactorCache) -> xs.Array:
        """
        Load the factor from the cache or compute and cache it before returning.
        """
        
        try:
            return cache[cls]
        except KeyError:
            result = cls._compute(x, cache)
            cache[cls] = result
            return result
    
    @classmethod
    @abstractmethod
    def _compute(cls, x: xs.CrossSection, cache: FactorCache) -> xs.Array:
        ...
