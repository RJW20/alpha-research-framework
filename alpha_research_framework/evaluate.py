import pandas as pd

from alpha_research_framework.alphas import Alpha
from alpha_research_framework.spearman_rank import spearman_rank
from alpha_research_framework.universe import Universe


def evaluate(universe: Universe, alphas: list[Alpha]) -> pd.DataFrame:
    """
    Evaluate cross-sectional predictive power of one or more alphas using
    Information Coefficient (IC) analysis.

    Returns a pd.DataFrame indexed by date, with one column per (alpha, horizon)
    pair, containing the cross-sectional correlation between alpha scores at
    time t and forward returns at time t + horizon
    Missing values may appear where:
    - alpha signals cannot be computed (start of sample)
    - forward returns cannot be computed (end of sample)
    - the cross-section is too small after filtering
    """

    tuples = [(a.NAME, h) for a in alphas for h in sorted(a.HORIZONS)]
    columns = pd.MultiIndex.from_tuples(tuples, names=["alpha", "horizons"])
    ic_df = pd.DataFrame(index=universe.dates, columns=columns, dtype=float)

    features = set().union(*[alpha.required_features for alpha in alphas])
    universe.build_features(features)
    for date in universe.dates:
        x = universe.cross_section(date)
        fut_ret = universe.future_returns(date)
        for alpha in alphas:
            signal = alpha.compute(x)
            for horizon in alpha.HORIZONS:
                ic_df.loc[date, (alpha.NAME, horizon)] = spearman_rank(
                    signal, fut_ret[horizon]
                )

    return ic_df
