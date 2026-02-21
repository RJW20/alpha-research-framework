from typing import Callable

import alpha_research_framework.market_data as md
from alpha_research_framework.evaluate.metrics.information_coefficient import (
    information_coefficient,
)
from alpha_research_framework.evaluate.metrics.metric import Metric

REGISTRY: dict[Metric, Callable[[md.Array, md.Array], float]] = {
    "ic": information_coefficient,
}
