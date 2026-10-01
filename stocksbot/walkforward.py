"""Walk-forward validation: pick parameters on a training window, then score
them only on the following unseen window, and roll forward. The stitched
out-of-sample equity is the honest estimate; in-sample "best params" results
are optimistic by construction.
"""
from __future__ import annotations

import itertools

import pandas as pd

from . import backtest as bt


def _grid(space: dict) -> list[dict]:
    if not space:
        return [{}]
    keys = list(space)
    return [dict(zip(keys, vals)) for vals in itertools.product(*space.values())]


def walk_forward(df, fn, space, train_years=4, test_years=1, costs=bt.Costs(), risk=bt.Risk()):
    days = bt.TRADING_DAYS
    train, test = train_years * days, test_years * days
    segments, chosen = [], []
    start = 0
    while start + train + test // 4 <= len(df):
        tr = df.iloc[start:start + train]
        best, best_sharpe = {}, float("-inf")
        for params in _grid(space):
            s = bt.metrics(bt.run(tr, fn(tr, **params), costs, risk))["sharpe"]
            if s > best_sharpe:
                best, best_sharpe = params, s
        # Compute the signal with history (indicators need warm-up) but only
        # score the unseen test slice.
        hist = df.iloc[: start + train + test]
        sig = fn(hist, **best)
        te_idx = hist.index[start + train:]
        res = bt.run(hist.loc[te_idx], sig.loc[te_idx], costs, risk)
        segments.append(res["strat_ret"])
        chosen.append({"test_start": te_idx[0].date(), "params": best, "train_sharpe": round(best_sharpe, 2)})
        start += test
    if not segments:
        raise ValueError("not enough history for one walk-forward split")
    r = pd.concat(segments)
    oos = pd.DataFrame({"strat_ret": r, "equity": (1 + r).cumprod(), "close": df["Close"].reindex(r.index)})
    oos["position"] = 0.0  # positions not stitched; exposure not reported for OOS
    return oos, chosen
