"""Run every strategy on every sample ticker, in-sample and walk-forward, and
write results/summary.csv, results/report.md and results/equity.png.

    python run_backtest.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from stocksbot import backtest as bt
from stocksbot.data import sample_universe
from stocksbot.strategies import STRATEGIES
from stocksbot.walkforward import walk_forward

OUT = Path(__file__).resolve().parent / "results"


def main():
    OUT.mkdir(exist_ok=True)
    universe = sample_universe()
    rows, oos_curves = [], {}
    for ticker, df in universe.items():
        for name, (fn, space) in STRATEGIES.items():
            ins = bt.metrics(bt.run(df, fn(df)))  # default params, full history
            oos, chosen = walk_forward(df, fn, space)
            m = bt.metrics(oos)
            rows.append({
                "ticker": ticker, "strategy": name,
                "start": df.index[0].date(), "end": df.index[-1].date(),
                "oos_cagr": m["cagr"], "oos_sharpe": m["sharpe"], "oos_max_dd": m["max_dd"],
                "full_cagr": ins["cagr"], "full_sharpe": ins["sharpe"], "full_max_dd": ins["max_dd"],
                "exposure": ins["exposure"], "best_day": ins["best_day"],
                "buy_hold_cagr": ins["buy_hold_cagr"], "buy_hold_max_dd": ins["buy_hold_max_dd"],
            })
            if name == "ensemble":
                oos_curves[ticker] = oos["equity"]
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "summary.csv", index=False, float_format="%.4f")

    agg = summary.groupby("strategy")[["oos_cagr", "oos_sharpe", "oos_max_dd", "buy_hold_cagr", "buy_hold_max_dd"]].median()
    agg = agg.sort_values("oos_sharpe", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    for t, eq in oos_curves.items():
        ax.plot(eq.index, eq.values, label=t, linewidth=1.2)
    ax.set_yscale("log"); ax.set_title("Ensemble strategy, walk-forward out-of-sample equity (log scale)")
    ax.set_ylabel("Growth of $1"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "equity.png", dpi=120)

    pct = lambda x: f"{x:.1%}"
    with open(OUT / "report.md", "w") as f:
        f.write("# Backtest results\n\n")
        f.write("Median across tickers, walk-forward out-of-sample (4y train / 1y test, rolled yearly), "
                "after 6 bps per side costs, 15% vol target, 20% drawdown kill switch.\n\n")
        f.write(agg.to_markdown(floatfmt=".2f") + "\n\n")
        f.write("## Per ticker\n\n")
        f.write(summary.to_markdown(index=False, floatfmt=".2f") + "\n")
    print(agg.to_string(float_format=lambda x: f"{x:.3f}"))
    print()
    print(summary[["ticker", "strategy", "oos_cagr", "oos_sharpe", "oos_max_dd", "buy_hold_cagr", "buy_hold_max_dd", "best_day"]]
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
