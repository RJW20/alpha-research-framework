from pathlib import Path

import numpy as np


class MarketData:
    """
    Small class holding np.memmaps of market data.
    
    price = adjusted close, split/dividend adjusted
    volume = adjusted volume, aligned to price
    """

    def __init__(self, path: Path, shape: tuple[int, int]) -> None:
        self.price = np.memmap(
            path / "price.dat",
            dtype=np.float32,
            mode="w+",
            shape=shape,
        )
        self.volume = np.memmap(
            path / "volume.dat",
            dtype=np.float32,
            mode="w+",
            shape=shape,
        )

    def __getitem__(self, key):
        return self.price[key], self.volume[key]
    
    def __setitem__(self, key, value) -> None:
        self.price[key], self.volume[key] = value

    def flush(self) -> None:
        self.price.flush()
        self.volume.flush()
