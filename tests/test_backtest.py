import json

import numpy as np
import pandas as pd
import pytest

from stocksbot import backtest as bt
from stocksbot.strategies import STRATEGIES
from stocksbot.webhook import parse_alert


def _df(close):
    idx = pd.bdate_range("2020-01-01", periods=len(close))
    c = pd.Series(close, index=idx, dtype=float)
    return pd.DataFrame({"Open": c, "High": c, "Low": c, "Close": c, "Volume": 1.0})


def test_no_lookahead():
    # A signal built from today's return must only earn tomorrow's return.
    rng = np.random.default_rng(0)
    df = _df(100 * np.cumprod(1 + rng.normal(0, 0.01, 2000)))
    today_up = (df["Close"].pct_change() > 0).astype(float)
    res = bt.run(df, today_up, bt.Costs(0, 0), bt.Risk(target_vol=None, max_dd=None))
    assert abs(bt.metrics(res)["sharpe"]) < 1.0  # random walk: no edge
    # Sanity: an oracle that peeks at tomorrow is (rightly) credited.
    oracle = (df["Close"].pct_change().shift(-1) > 0).astype(float)
    res = bt.run(df, oracle, bt.Costs(0, 0), bt.Risk(target_vol=None, max_dd=None))
    assert bt.metrics(res)["sharpe"] > 5


def test_flat_signal_is_flat():
    df = _df(np.linspace(100, 200, 300))
    res = bt.run(df, pd.Series(0.0, index=df.index))
    assert res["equity"].iloc[-1] == pytest.approx(1.0)


def test_costs_charged():
    df = _df([100.0] * 10)
    flip = pd.Series([0, 1] * 5, index=df.index, dtype=float)
    res = bt.run(df, flip, bt.Costs(fee_bps=10, slippage_bps=0), bt.Risk(target_vol=None, max_dd=None))
    assert res["equity"].iloc[-1] == pytest.approx((1 - 0.001) ** 9)


def test_kill_switch_limits_drawdown():
    df = _df(np.r_[np.full(10, 100.0), np.linspace(100, 20, 100)])
    res = bt.run(df, pd.Series(1.0, index=df.index), bt.Costs(0, 0), bt.Risk(target_vol=None, max_dd=0.2, cooldown=1000))
    assert bt.metrics(res)["max_dd"] > -0.25


@pytest.mark.parametrize("name", list(STRATEGIES))
def test_strategies_bounded(name):
    rng = np.random.default_rng(1)
    df = _df(100 * np.cumprod(1 + rng.normal(0.0003, 0.015, 800)))
    sig = STRATEGIES[name][0](df)
    assert sig.between(0, 1).all()


def test_webhook_rejects_bad_secret():
    with pytest.raises(PermissionError):
        parse_alert(json.dumps({"secret": "x", "ticker": "SPY", "action": "buy"}).encode(), "y")
    a = parse_alert(json.dumps({"secret": "y", "ticker": "SPY", "action": "buy"}).encode(), "y")
    assert a["ticker"] == "SPY"
