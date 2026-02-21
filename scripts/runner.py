from pathlib import Path

from alpha_research_framework import Alpha, EquityData, Universe, Window, evaluate
from alpha_research_framework.alphas import (
    Momentum12To1,
    Reversal1d,
    Reversal5d,
    RiskAdjustedReturns20d,
    RiskAdjustedMomentum12To1,
    Volatility20d,
    Volatility12,
)
from alpha_research_framework.alphas.returns_based import ReturnsBased

SRC = Path("data")
PATH = Path("universe")


class Returns20d(ReturnsBased):

    NAME = "returns_20d"
    CATEGORY = "momentum/trend"

    LOOKBACK = Window.MONTH
    HORIZONS = {Window.DAY, Window.WEEK, Window.MONTH}


def run() -> None:

    equity_data = EquityData(SRC)
    u = Universe(PATH, equity_data, 1e6, 1e8)
    alphas: list[Alpha] = [
        Reversal1d(), Reversal5d(),
        Returns20d(),
        Momentum12To1(),
        Volatility20d(), Volatility12(),
        RiskAdjustedReturns20d(), RiskAdjustedMomentum12To1()
    ]
    result = evaluate(u, alphas)
    ic_df = result["ic"]
    print(ic_df.mean())


if __name__ == "__main__":
    run()
