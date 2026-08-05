from typing import ParamSpec, Protocol

import numpy as np
import numpy.typing as npt

P = ParamSpec("P")


class TransformFunc(Protocol[P]):
    """
    Function signature for all market series transforms.

    Only acts over the time axis (axis 0).
    Carries out the transformation in-place.
    """

    @property
    def __name__(self) -> str:
        ...

    def __call__(
        self, arr: npt.NDArray[np.floating],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> None:
        ...
