from pathlib import Path

import numpy as np

import alpha_research_framework.market_data as md


class MarketDataStore:
    """
    Disk-backed market data storage.
    
    `price` = adjusted close
    `volume` = adjusted volume
    """

    def __init__(self, path: Path, shape: tuple[int, int]) -> None:
        self.price = np.memmap(
            path / "price.dat",
            dtype=md.Scalar,
            mode="w+",
            shape=shape,
        )
        self.volume = np.memmap(
            path / "volume.dat",
            dtype=md.Scalar,
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
