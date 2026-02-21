import alpha_research_framework.market_data as md
from alpha_research_framework.evaluate.metrics.information_coefficient.spearman_rank import (  # noqa: E501
    spearman_rank,
)


def information_coefficient(signal: md.Array, future_returns: md.Array) -> float:
    """
    Return the Spearman's rank correlation coefficient between signal and
    future_returns.
    """
    return spearman_rank(signal, future_returns)
