from functools import partial
from typing import Any

from .series import Series
from .transformed_series import TransformedSeries
from .transforms import TransformFunc


def transform(
    source: type[Series],
    *,
    target: bool = False,
    func: TransformFunc,
    **kwargs: Any,
) -> type[TransformedSeries]:
    """
    Return a dynamically generated concrete subclass of `TransformedSeries`.

    The subclass' `TAG` is `PREDICTOR` unless `target=True`.
    The subclass' `SOURCE` is `source`.
    The subclass' `TRANSFORM` is a partial function ready to act on an
    `md.Array` with `kwargs` forwarded.
    """

    if not issubclass(source, Series):                                          # type: ignore
        raise TypeError(
            f"Unable to build new series: {source.__name__} must be a subclass "
            "Series"
        )

    name = (
        f"{func.__name__}({source.__name__}"
        f"{",".join(f",{k}={v}" for k, v in kwargs.items())})"
    )

    if not target:
        if source.TAG == Series.Tag.TARGET:
            raise ValueError(
                f"Unable to build new series: cannot tag {name} as a predictor "
                f"when it is derived from {source.__name__} which is tagged as "
                "a target"
            )
        tag = Series.Tag.PREDICTOR
    else:
        tag = Series.Tag.TARGET

    return type(
        name,
        (TransformedSeries,),
        {
            "TAG": tag,
            "SOURCE": source,
            "TRANSFORM": partial(func, **kwargs),
        }
    )
