from pathlib import Path

import numpy as np


class MarketData:
    """Small class holding np.memmaps for market data."""

    def __init__(self, path: Path, shape: tuple[int, int]) -> None:
        self.adj_close = np.memmap(
            path / "adj_close.dat",
            dtype=np.float32,
            mode="w+",
            shape=shape,
        )
        self.adj_volume = np.memmap(
            path / "adj_volume.dat",
            dtype=np.float32,
            mode="w+",
            shape=shape,
        )

    def flush(self) -> None:
        """Flush all memmaps."""

        self.adj_close.flush()
        self.adj_volume.flush()
