"""Signal layer. Each strategy maps an OHLCV frame to a target position in [0, 1]
(long-only, no leverage) decided at the bar's close. The backtester shifts it
one bar, so a signal can never trade on the data that produced it.

Ideas borrowed (re-implemented, no GPL code copied):
  sma_cross      classic trend filter (freqtrade/jesse sample strategies)
  donchian       Turtle-style channel breakout (backtrader examples)
  rsi_reversion  Connors-style short-term mean reversion in an uptrend
  bollinger      band mean reversion (freqtrade BBRSI-style)
  tsmom          12-month time-series momentum (academic, Lean alpha models)
  ensemble       vote of the above, Lean-style alpha combination
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi(close: pd.Series, n: int) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / down.replace(0, np.nan))


def _hold(enter: pd.Series, exit_: pd.Series) -> pd.Series:
    """Turn entry/exit events into a 0/1 holding series."""
    state = pd.Series(np.nan, index=enter.index)
    state[enter] = 1.0
    state[exit_ & ~enter] = 0.0
    return state.ffill().fillna(0.0)


def sma_cross(df: pd.DataFrame, fast: int = 50, slow: int = 200) -> pd.Series:
    c = df["Close"]
    return (c.rolling(fast).mean() > c.rolling(slow).mean()).astype(float)


def donchian(df: pd.DataFrame, entry: int = 55, exit: int = 20) -> pd.Series:
    c = df["Close"]
    enter = c > c.rolling(entry).max().shift(1)
    exit_ = c < c.rolling(exit).min().shift(1)
    return _hold(enter, exit_)


def rsi_reversion(df: pd.DataFrame, n: int = 2, buy_below: float = 10, trend: int = 200) -> pd.Series:
    c = df["Close"]
    r = _rsi(c, n)
    uptrend = c > c.rolling(trend).mean()
    enter = (r < buy_below) & uptrend
    exit_ = (c > c.rolling(5).mean()) | ~uptrend
    return _hold(enter, exit_)


def bollinger(df: pd.DataFrame, n: int = 20, k: float = 2.0, trend: int = 200) -> pd.Series:
    c = df["Close"]
    mid, sd = c.rolling(n).mean(), c.rolling(n).std()
    enter = (c < mid - k * sd) & (c > c.rolling(trend).mean())
    exit_ = c > mid
    return _hold(enter, exit_)


def tsmom(df: pd.DataFrame, lookback: int = 252, skip: int = 21) -> pd.Series:
    c = df["Close"]
    return (c.shift(skip) / c.shift(lookback) - 1 > 0).astype(float)


def ensemble(df: pd.DataFrame, members=None) -> pd.Series:
    members = members or [sma_cross, donchian, rsi_reversion, bollinger, tsmom]
    return pd.concat([m(df) for m in members], axis=1).mean(axis=1)


STRATEGIES = {
    "sma_cross": (sma_cross, {"fast": [20, 50], "slow": [100, 200]}),
    "donchian": (donchian, {"entry": [20, 55], "exit": [10, 20]}),
    "rsi_reversion": (rsi_reversion, {"n": [2, 3], "buy_below": [5, 10, 20]}),
    "bollinger": (bollinger, {"n": [20], "k": [1.5, 2.0, 2.5]}),
    "tsmom": (tsmom, {"lookback": [126, 252]}),
    "ensemble": (ensemble, {}),
}
