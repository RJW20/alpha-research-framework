from typing import Any, Callable, TypeAlias

import alpha_research_framework.market_data as md

TransformFunc: TypeAlias = \
    Callable[[md.Array], None] | Callable[[md.Array, Any], None]
"""
Function signature for all market series transforms.

Only acts over the time axis (axis 0).
Carries out the transformation in-place.
"""
