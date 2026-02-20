import pandas as pd

from alpha_research_framework.alphas import Alpha
from alpha_research_framework.evaluate.spearman_rank import spearman_rank
from alpha_research_framework.universe import Universe
from alpha_research_framework.window import Window


def evaluate(universe: Universe, alphas: list[Alpha]) -> pd.DataFrame:
    """
    Evaluate cross-sectional predictive power of one or more alphas using
    Information Coefficient (IC) analysis.

    Returns a pd.DataFrame indexed by date, with one column per (alpha, horizon)
    pair, containing the cross-sectional correlation between alpha scores at
    time t and forward returns between times t + 1 and t + horizon.
    Missing values will appear where:
    - frequency does not align with horizon
    - alpha signals cannot be computed (start of sample)
    - forward returns cannot be computed (end of sample)
    - the cross-section is too small after filtering
    """

    tuples = [(a.NAME, h) for a in alphas for h in sorted(a.HORIZONS)]
    columns = pd.MultiIndex.from_tuples(tuples, names=["alpha", "horizons"])
    ic_df = pd.DataFrame(index=universe.dates, columns=columns, dtype=float)

    features = set().union(*[alpha.required_features for alpha in alphas])
    universe.build_features(features)
    horizons = {h for a in alphas for h in a.HORIZONS}
    for t, date in enumerate(universe.dates):

        valid_horizons = {w for w in Window if not t % w.value} & horizons
        if not valid_horizons:
            continue

        x = universe.cross_section(date)
        fut_ret = universe.future_returns(date)
        for alpha in alphas:
            horizons_to_evaluate = valid_horizons & alpha.HORIZONS
            if not horizons_to_evaluate:
                continue
            signal = alpha.compute(x)
            for horizon in horizons_to_evaluate:
                ic_df.loc[date, (alpha.NAME, horizon)] = spearman_rank(
                    signal, fut_ret[horizon]
                )

    return ic_df
