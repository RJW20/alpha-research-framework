import pandas as pd

from alpha_research_framework.alphas import Alpha
from alpha_research_framework.evaluate.metrics import Metric, compute_metric
from alpha_research_framework.universe import Universe
from alpha_research_framework.window import Window


def evaluate(
    universe: Universe, alphas: list[Alpha], metrics: set[Metric] = {"ic"}
) -> dict[Metric, pd.DataFrame]:
    """
    Evaluate cross-sectional predictive power of one or more alphas.

    Returns a dictionary mapping metrics to pd.DataFrames indexed by date, with
    one column per (alpha, horizon) pair, containing a measure of the
    correlation between alpha scores at time t and returns between times t and
    t + horizon:
    - `ic` - Information Coefficient: Spearman's rank correlation coefficient
    between signal and future returns.

    Missing values will appear where:
    - frequency does not align with horizon
    - alpha signals cannot be computed (start of sample)
    - forward returns cannot be computed (end of sample)
    - the cross-section is too small after filtering
    """

    tuples = [(a.NAME, h) for a in alphas for h in sorted(a.HORIZONS)]
    columns = pd.MultiIndex.from_tuples(tuples, names=["alpha", "horizon"])
    result: dict[Metric, pd.DataFrame] = dict()
    for metric in metrics:
        result[metric] = pd.DataFrame(
            index=universe.dates,
            columns=columns,
            dtype=float
        )

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
                for metric in metrics:
                    result[metric].loc[date, (alpha.NAME, horizon)] = (
                        compute_metric(metric, signal, fut_ret[horizon])
                    )

    return result
