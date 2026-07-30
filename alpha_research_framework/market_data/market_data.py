from typing import TypeAlias

import alpha_research_framework.observables as observables

from .array import Array

MarketData: TypeAlias = dict[type[observables.Observable], Array]
"""
3D representation of the stock market.

Implemented as a mapping of `Observable` to `Array`.
"""
