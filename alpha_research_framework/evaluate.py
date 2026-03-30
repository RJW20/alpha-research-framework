from typing import Sequence

import pandas as pd

import alpha_research_framework.alphas as alphas
import alpha_research_framework.features as features
import alpha_research_framework.metrics as metrics
from alpha_research_framework.registrable import resolve
from alpha_research_framework.universe import Universe
from alpha_research_framework.window import Window


def _create_dataframes(
    alphas_: list[type[alphas.Alpha]],
    metrics_: list[type[metrics.Metric]],
    index: pd.Index,
) -> dict[str, pd.DataFrame]:
    """
    Return a dictionary mapping each `alpha.ID` to an empty `DataFrame` with
    given index and columns per `(horizon, metric.ID, metric.MEASURES)` for
    every horizon required by the alpha, for every metric in `metrics` and for
    each of a metrics measures if it is multi-valued.
    """

    result: dict[str, pd.DataFrame] = dict()

    for alpha in alphas_:
        tuples: list[tuple[str | Window,...]] = list()
        for horizon in sorted(alpha.HORIZONS):
            for metric in metrics_:
                if issubclass(metric, metrics.MultiValueMetric):
                    tuples += [
                        (horizon, metric.ID, measure)
                        for measure in metric.MEASURES
                    ]
                else:
                    tuples.append((horizon, metric.ID, ""))
        names = ["horizon", "metric", "measures"]
        columns = pd.MultiIndex.from_tuples(tuples, names=names)
        result[alpha.ID] = pd.DataFrame(
            index=index,
            columns=columns,
            dtype=float,
        )

    return result


def _evaluate(
    universe: Universe,
    alphas_: list[type[alphas.Alpha]],
    metrics_: list[type[metrics.Metric]],
    dfs: dict[str, pd.DataFrame],
) -> None:
    """
    Populate the provided `DataFrame`s with metric evaluations for each alpha
    for each date in `universe`.

    Iterates through the universe dates, computing alpha signals and
    corresponding forward returns over in-phase horizons, and writes the
    resulting metric values into `dfs` in-place.
    """

    window_to_forward_returns = {
        Window.DAY:         features.ForwardReturns1d,
        Window.WEEK:        features.ForwardReturns5d,
        Window.MONTH:       features.ForwardReturns20d,
        Window.QUARTER:     features.ForwardReturns63d,
        Window.HALF_YEAR:   features.ForwardReturns126d,
        Window.YEAR:        features.ForwardReturns252d,
    }

    horizons = {h for a in alphas_ for h in a.HORIZONS}
    for t, date in enumerate(universe.dates):

        horizons_in_phase = {w for w in Window if not t % w.value} & horizons
        if not horizons_in_phase:
            continue

        xs, forward_returns = universe[date]
        cache = alphas.factors.FactorCache()
        for alpha in alphas_:

            horizons_to_evaluate = horizons_in_phase & alpha.HORIZONS
            if not horizons_to_evaluate:
                continue

            df = dfs[alpha.ID]
            signal = alpha.compute(xs, cache)
            for horizon in horizons_to_evaluate:
                forward_returns_over_horizon = (
                    forward_returns[window_to_forward_returns[horizon]]
                )
                for metric in metrics_:
                    df.loc[date, pd.IndexSlice[horizon, metric.ID, :]] = (
                        metric.compute(signal, forward_returns_over_horizon)
                    )


def evaluate(
    universe: Universe,
    alphas_: Sequence[str | type[alphas.Alpha]],
    metrics_: Sequence[str | type[metrics.Metric]] = ["information_coefficient"]
) -> dict[str, pd.DataFrame]:
    """
    Evaluate the cross-sectional predictive power of one or more alphas.

    Computes cross-sectional metrics that quantify the relationship between
    each alpha's signal at time `t` and the realised forward returns from
    `t` to `t + horizon`.

    Parameters
    ----------
    universe : Universe
        Cross-sectional universe with pre-applied (optional) sector/industry,
        liquidity and market-cap filtering.
    
    alphas_ : Sequence[str | type[Alpha]]
        List of alphas to evaluate, given by either `ID` or type name. Available
        options include:
        - `"reversal_1d"` or `Reversal1d`
        - `"momentum_12m_1m"` or `Momentum12m1m`
        - `"volatility_20d"` or `Volatility20d`

        For a full list of available alphas along with information on creating
        custom alphas see the "Alphas" section of the documentation.
    
    metrics_ : Sequence[str | type[Metric]], default ["information_coefficient"]
        List of metrics to compute per alpha at every `t`, given by either `ID`
        or type name. Available options are:
        - `"information_coefficient"` or `InformationCoefficient`
        - `"quantile_portfolio"` or `QuantilePorfolio`

        For information on creating custom metrics see the "Metrics" section of
        the documentation.

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary mapping `alpha.ID` to a `DataFrame` indexed by date, with
        columns per `(horizon, metric)` pair, containing a measure of the
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

    TypeError
        If a `type` in `alphas` is not a subclass of `Alpha` or a `type` in
        `metrics` is not a subclass of `Metric`.
    """

    resolved_alphas = list(
        dict.fromkeys([resolve(a, alphas.Alpha) for a in alphas_])
    )
    resolved_metrics = list(
        dict.fromkeys([resolve(m, metrics.Metric) for m in metrics_])
    )
    result = _create_dataframes(
        resolved_alphas,
        resolved_metrics,
        index=universe.dates,
    )
    _evaluate(universe, resolved_alphas, resolved_metrics, result)

    return result
