from collections.abc import Callable
from typing import Any, TypeAlias

import numpy as np
import numpy.typing as npt

TransformFunc: TypeAlias = \
    Callable[[npt.NDArray[np.floating]], None] | \
    Callable[[npt.NDArray[np.floating], Any], None]
"""
Function signature for all market series transforms.

Only acts over the time axis (axis 0).
Carries out the transformation in-place.
"""
