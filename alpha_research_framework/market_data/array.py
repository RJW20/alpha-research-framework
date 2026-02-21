from typing import TypeAlias

import numpy.typing as npt

from alpha_research_framework.market_data.scalar import Scalar

Array: TypeAlias = npt.NDArray[Scalar]
"""Format for all stores of market-like data (raw and features)."""
