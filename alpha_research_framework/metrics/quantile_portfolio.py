from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

import alpha_research_framework.market_data as md
import alpha_research_framework.metrics.stats as stats
from alpha_research_framework.metrics.metric import Metric


class QuantilePorfolio(Metric):

    __quantiles__ = 10

    ID = "qp"

    @staticmethod
    def dataframe(
        index: pd.Index,
        base_columns: list[tuple[Any,...]],
        base_column_names: list[str]
    ) -> pd.DataFrame:
        deciles = [f"Q{i}" for i in range(10)]
        tuples = [
            (*bc, dec)
            for bc in base_columns
            for dec in deciles
        ]
        columns = pd.MultiIndex.from_tuples(
            tuples=tuples,
            names=base_column_names + ["decile"]
        )
        return pd.DataFrame(
            index=index,
            columns=columns,
            dtype=float
        )
    
    @staticmethod
    def compute(
        signal: md.Array,
        future_returns: md.Array
    ) -> npt.NDArray[np.floating]:
        """
        Return a numpy array of average future returns per quantile in signal.
        """

        valid = stats.extract_valid(signal, future_returns)
        if not valid:
            return np.full(QuantilePorfolio.__quantiles__, np.nan)
        signal_clean, future_returns_clean = valid
        
        q_idx = stats.quantile_indices(
            signal_clean,
            QuantilePorfolio.__quantiles__
        )
        q_avg = stats.bucket_averages(
            q_idx,
            future_returns_clean,
            QuantilePorfolio.__quantiles__
        )
        return q_avg
