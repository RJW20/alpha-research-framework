from enum import Enum, auto


class FeatureTag(Enum):
    """Enum for tagging feature types."""

    PREDICTOR = auto()
    TARGET = auto()
