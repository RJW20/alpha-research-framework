from typing import Any, override

import alpha_research_framework.market_data as md
from alpha_research_framework.features.derived_feature import DerivedFeature
from alpha_research_framework.features.feature import Feature
from alpha_research_framework.features.feature_cache import FeatureCache
from alpha_research_framework.operator import OperatorMeta


class TransformMeta(OperatorMeta):
    """
    Metaclass for feature transforms.

    Provides utility for applying transforms to features.
    """

    @override
    def __call__(
        cls,                                                                    # noqa: N805
        feature_cls: type[Feature],
        *,
        target: bool = False,
        **kwargs: Any,
    ) -> type[DerivedFeature]:
        """
        Return a dynamicly generated concrete subclass of `DerivedFeature`.

        The subclass' `TAG` will be `PREDICTOR` unless `target=True`.
        The subclass' `SOURCE` will be `feature_cls`.
        The subclass' `compute` will be the application of `cls.compute` to the
        result of `feature_cls.compute`.
        `kwargs` are forwarded to `cls.compute`.
        """

        if not issubclass(feature_cls, Feature):                                # type: ignore
            raise TypeError(
                f"Unable to build new feature: {feature_cls.__name__} must be "
                "a subclass of Feature"
            )

        name = f"{cls.__name__}({feature_cls.__name__})"

        if not target:
            if feature_cls.TAG == Feature.Tag.TARGET:
                raise ValueError(
                    f"Unable to build new feature: cannot tag {name} as a "
                    f"predictor when it is derived from {feature_cls.__name__} "
                    "which is tagged as a target"
                )
            tag = Feature.Tag.PREDICTOR
        else:
            tag = Feature.Tag.TARGET

        source = feature_cls

        def compute(
            _cls: type[DerivedFeature],
            market_data: md.MarketData,
            cache: FeatureCache,
            out: md.Array,
        ) -> None:
            feature_cls.compute(market_data, cache, out)
            cls.compute(out, **kwargs)                                          # type: ignore[attr-defined]

        return type(
            name,
            (DerivedFeature,),
            {
                "TAG": tag,
                "SOURCE": source,
                "compute": classmethod(compute)
            }
        )
