"""Vectorized daily backtester with costs and risk controls.

Execution model: the target from bar t's close is held from bar t+1 (we trade at
t's close in effect, earning t+1's close-to-close return). Every change in
position pays fee + slippage on the traded notional.

Risk layer:
  * volatility targeting: scale exposure so expected annual vol ~= target_vol
  * max_exposure caps the scale (1.0 = no leverage)
  * drawdown kill switch: if equity falls max_dd below its peak, go flat for
    cooldown bars, then resume from the new (lower) peak
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass
class Costs:
    fee_bps: float = 1.0       # commission / exchange fees per side
    slippage_bps: float = 5.0  # spread + impact per side

    @property
    def per_side(self) -> float:
        return (self.fee_bps + self.slippage_bps) / 1e4


@dataclass
class Risk:
    target_vol: float | None = 0.15
    vol_lookback: int = 20
    max_exposure: float = 1.0
    max_dd: float | None = 0.20
    cooldown: int = 20


def run(df: pd.DataFrame, signal: pd.Series, costs: Costs = Costs(), risk: Risk = Risk()) -> pd.DataFrame:
    close = df["Close"]
    ret = close.pct_change().fillna(0.0)
    target = signal.reindex(close.index).fillna(0.0).clip(0, 1)

    if risk.target_vol:
        vol = ret.rolling(risk.vol_lookback).std() * np.sqrt(TRADING_DAYS)
        scale = (risk.target_vol / vol).clip(upper=risk.max_exposure).fillna(0.0)
    else:
        scale = pd.Series(risk.max_exposure, index=close.index)
    target = (target * scale).clip(0, risk.max_exposure)

    # Kill switch needs path dependency, so loop (fast enough for daily bars).
    pos = np.zeros(len(close))
    equity = np.ones(len(close))
    tgt, r = target.to_numpy(), ret.to_numpy()
    peak, halted_until, prev = 1.0, -1, 0.0
    for i in range(1, len(close)):
        held = prev
        eq = equity[i - 1] * (1 + held * r[i])
        want = tgt[i] if i > halted_until else 0.0
        if risk.max_dd and i > halted_until and eq < peak * (1 - risk.max_dd):
            want, halted_until, peak = 0.0, i + risk.cooldown, eq
        cost = abs(want - held) * costs.per_side
        equity[i] = eq * (1 - cost)
        peak = max(peak, equity[i]) if i > halted_until else peak
        pos[i] = want
        prev = want
    out = pd.DataFrame({"close": close, "position": pos, "equity": equity}, index=close.index)
    out["strat_ret"] = out["equity"].pct_change().fillna(0.0)
    return out


def metrics(res: pd.DataFrame) -> dict:
    eq, r = res["equity"], res["strat_ret"]
    years = max(len(res) / TRADING_DAYS, 1e-9)
    cagr = eq.iloc[-1] ** (1 / years) - 1
    sharpe = r.mean() / r.std() * np.sqrt(TRADING_DAYS) if r.std() > 0 else 0.0
    dd = (eq / eq.cummax() - 1).min()
    bh = res["close"].iloc[-1] / res["close"].iloc[0]
    bh_dd = (res["close"] / res["close"].cummax() - 1).min()
    trades = int((res["position"].diff().abs() > 1e-9).sum())
    return {
        "cagr": cagr,
        "sharpe": sharpe,
        "max_dd": dd,
        "total_return": eq.iloc[-1] - 1,
        "exposure": res["position"].gt(0).mean(),
        "rebalances": trades,
        "buy_hold_cagr": bh ** (1 / years) - 1,
        "buy_hold_max_dd": bh_dd,
        "best_day": r.max(),
    }
