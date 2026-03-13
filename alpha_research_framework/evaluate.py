import pandas as pd

from alpha_research_framework.alphas import Alpha
from alpha_research_framework.metrics import Metric, MultiValueMetric
from alpha_research_framework.operator import Operator
from alpha_research_framework.universe import Universe
from alpha_research_framework.window import Window


def _resolve(
    operator: str | type[Operator],
    operator_subtype: type[Operator]
) -> type[Operator]:
    
    if isinstance(operator, str):
        return operator_subtype.from_id(operator)
    return operator


def evaluate(
    universe: Universe,
    alphas: list[str | type[Alpha]] = ["reversal_1d"],
    metrics: list[str | type[Metric]] = ["information_coefficient"],
) -> dict[str, pd.DataFrame]:
    """
    Evaluate the cross-sectional predictive power of one or more alphas.

    Computes cross-sectional metrics that quantify the relationship between
    each alpha's signal at time `t` and the realised forward returns from
    `t` to `t + horizon`.

    Parameters
    ----------
    universe : Universe
        Cross-sectional universe with pre-applied liquidity and market-cap
        filters along with optional sector/industry filters applied via its
        underlying `EquityData`.
    
    alphas : list[str | type[Alpha]], default ["reversal_1d"]
        List of alphas to evaluate, given by either `ID` or concrete subclass
        type name. Available options include:
        - `"reversal_1d"` or `Reversal1d`
        - `"momentum_12m_1m"` or `Momentum12m1m`
        - `"volatility_20d"` or `Volatility20d`

        For a full list of available alphas along with information on creating
        custom alphas see the "Alphas" section of the documentation.
    
    metrics : list[str | type[Metric]], default ["information_coefficient"]
        List of metrics to compute per alpha at every `t`, given by either `ID`
        or concrete subclass type name. Available options are:
        - `"information_coefficient"` or `InformationCoefficient`
        - `"quantile_portfolio"` or `QuantilePorfolio`

        For information on creating custom metrics see the "Metrics" section of
        the documentation.

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary mapping `metric.ID` to a `DataFrame` indexed by date, with
        columns per `(alpha, horizon)` pair, containing a measure of the
        correlation between each alpha's signal at time `t` and the realised
        forward returns from `t` to `t + horizon`. Missing values will appear in
        a `DataFrame` where:
        - Frequency does not align with horizon.
        - Alpha signals cannot be computed (start of sample).
        - Forward returns cannot be computed (end of sample).
        - Cross-section is too small after filtering.

    Raises
    ------
    ValueError
        If a `str` in `alphas` or `metrics` is unrecognised.
    """

    alphas = [_resolve(a, Alpha) for a in alphas]
    metrics = [_resolve(m, Metric) for m in metrics]

    base_tuples = [(a.ID, h) for a in alphas for h in sorted(a.HORIZONS)]
    base_names = ["alpha", "horizon"]
    result: dict[str, pd.DataFrame] = dict()
    for metric in metrics:

        if issubclass(metric, MultiValueMetric):
            tuples = [
                (*bt, m)
                for bt in base_tuples
                for m in metric.MEASURES
            ]
            names = base_names + [metric.MEASURE_GROUP]
        else:
            tuples = base_tuples
            names = base_names

        columns = pd.MultiIndex.from_tuples(tuples, names=names)
        result[metric.ID] = pd.DataFrame(
            index=universe.dates,
            columns=columns,
            dtype=float,
        )

    features = set().union(*[a.DEPENDENCIES for a in alphas])
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
                    df = result[metric.ID]
                    df.loc[date, pd.IndexSlice[alpha.ID, horizon, :]] = (
                        metric.compute(signal, fut_ret[horizon])
                    )

    return result
