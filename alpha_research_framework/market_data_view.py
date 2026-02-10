from typing import Protocol, TypeAlias, Union

import numpy as np
from numpy.typing import NDArray

MarketArray: TypeAlias = Union[NDArray[np.float32], np.memmap]


class MarketDataView(Protocol):
    """View over time * stock market data."""

    @property
    def price(self) -> MarketArray:
        ...

    @property
    def volume(self) -> MarketArray:
        ...
