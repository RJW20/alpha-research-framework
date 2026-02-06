from typing import TypeAlias, Union

import numpy as np
from numpy.typing import NDArray

MarketArray: TypeAlias = Union[NDArray[np.float32], np.memmap]
