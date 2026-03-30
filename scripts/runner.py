from pathlib import Path

import pandas as pd

import alpha_research_framework as arf

SRC = Path("data")
PATH = Path("universe")


def run() -> None:

    pd.set_option('display.max_rows', None)

    alphas = [
        "reversal_1d", "reversal_5d",
        "momentum_12m_1m", "risk_adjusted_momentum_12m_1m",
        "volatility_20d",
    ]
    metrics = ["information_coefficient"]
    u = arf.build_universe_for(alphas, src=SRC, path=PATH)
    result = arf.evaluate(u, alphas, metrics)
    for alpha in alphas:
        print(alpha, '\n', result[alpha].mean())


if __name__ == "__main__":
    run()
