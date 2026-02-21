import alpha_research_framework.market_data as md
from alpha_research_framework.evaluate.metrics.metric import Metric
from alpha_research_framework.evaluate.metrics.registry import REGISTRY


def compute_metric(
    metric: Metric,
    signal: md.Array,
    future_returns: md.Array
) -> float:
    """
    Return a value containing a measure of correlation between the signal and
    future_returns.
    
    - 'ic': information coefficient
    """

    return REGISTRY[metric](signal, future_returns)
