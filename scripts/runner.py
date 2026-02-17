from pathlib import Path

from alpha_research_framework import Alpha, EquityData, Universe, evaluate
from alpha_research_framework.alphas import (
    Momentum12To1,
    Reversal1d,
    Volatility20d,
)

SRC = Path("data")
PATH = Path("universe")


def run() -> None:

    equity_data = EquityData(SRC)
    u = Universe(PATH, equity_data, 0, 0)
    alphas: list[Alpha] = [Reversal1d(), Momentum12To1(), Volatility20d()]
    ic_df = evaluate(u, alphas)
    print(ic_df.mean())


if __name__ == "__main__":
    run()
