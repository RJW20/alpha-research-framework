from typing import Protocol, TypeAlias, Union

import numpy as np
import numpy.typing as npt

MarketArray: TypeAlias = Union[npt.NDArray[np.float32], np.memmap]


class MarketDataView(Protocol):
    """View over time * stock market data."""

    @property
    def price(self) -> MarketArray:
        ...

    @property
    def volume(self) -> MarketArray:
        ...
