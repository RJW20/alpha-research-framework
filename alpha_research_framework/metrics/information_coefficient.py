from typing import Any

import pandas as pd

import alpha_research_framework.market_data as md
from alpha_research_framework.metrics.metric import Metric
from alpha_research_framework.metrics.stats import spearman_rank


class InformationCoefficient(Metric):

    ID = "ic"

    @staticmethod
    def dataframe(
        index: pd.Index,
        base_columns: list[tuple[Any,...]],
        base_column_names: list[str]
    ) -> pd.DataFrame:
        columns = pd.MultiIndex.from_tuples(
            tuples=base_columns,
            names=base_column_names
        )
        return pd.DataFrame(
            index=index,
            columns=columns,
            dtype=float
        )
    
    @staticmethod
    def compute(signal: md.Array, future_returns: md.Array) -> float:
        """
        Return the Spearman's rank correlation coefficient between signal and
        future_returns.
        """
        return spearman_rank(signal, future_returns)
