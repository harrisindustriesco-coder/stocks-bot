"""Data layer: every source here is one whose terms allow automated use.

Finviz and TradingView are deliberately not scraped (both forbid it in their
terms). Supported legal routes:
  * Local CSVs (OHLCV or close-only), e.g. public datasets or broker exports.
  * Alpaca market data API (free account; keys read from env on your machine).
  * Finviz Elite screener export: you download the CSV with your own Elite
    subscription and point load_finviz_export() at it.
  * TradingView: send Pine Script webhook alerts to the bot (see webhook.py);
    no TradingView data is pulled.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def load_ohlcv_csv(path: str | Path) -> pd.DataFrame:
    """Load a Yahoo-style CSV (Date,Open,High,Low,Close,Adj Close,Volume).

    Prices are rescaled by Adj Close / Close so splits and dividends don't
    show up as fake crashes.
    """
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date").sort_index()
    df = df[~df.index.duplicated()].dropna()
    if "Adj Close" in df.columns:
        factor = df["Adj Close"] / df["Close"]
        for col in ("Open", "High", "Low", "Close"):
            df[col] = df[col] * factor
        df = df.drop(columns="Adj Close")
    return df[["Open", "High", "Low", "Close", "Volume"]]


def load_close_matrix(path: str | Path, date_col: str = "Date") -> dict[str, pd.DataFrame]:
    """Load a wide close-only CSV (one column per ticker) into per-ticker frames."""
    wide = pd.read_csv(path, parse_dates=[date_col], index_col=date_col).sort_index()
    out = {}
    for ticker in wide.columns:
        close = wide[ticker].dropna()
        out[ticker] = pd.DataFrame({"Open": close, "High": close, "Low": close,
                                    "Close": close, "Volume": float("nan")})
    return out


def sample_universe() -> dict[str, pd.DataFrame]:
    """Real daily US stock history from public GitHub datasets (see data/README.md)."""
    data = {
        "NVDA": load_ohlcv_csv(DATA_DIR / "nvda-1999-2014.csv"),
        "ORCL": load_ohlcv_csv(DATA_DIR / "orcl-1995-2014.csv"),
        "YHOO": load_ohlcv_csv(DATA_DIR / "yhoo-1996-2014.csv"),
    }
    for ticker, df in load_close_matrix(DATA_DIR / "stockdata-2007-2016.csv").items():
        data["SPX" if ticker == "GSPC" else ticker] = df
    return data


def load_finviz_export(path: str | Path) -> list[str]:
    """Tickers from a Finviz Elite screener CSV export you downloaded yourself."""
    return pd.read_csv(path)["Ticker"].dropna().astype(str).tolist()


def fetch_alpaca_daily(symbol: str, start: str, end: str | None = None) -> pd.DataFrame:
    """Daily bars from Alpaca's market data API.

    Needs ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY in your environment. Read-only
    market data; nothing here places orders.
    """
    import requests

    key, secret = os.environ["ALPACA_API_KEY_ID"], os.environ["ALPACA_API_SECRET_KEY"]
    params = {"timeframe": "1Day", "start": start, "adjustment": "all", "limit": 10000}
    if end:
        params["end"] = end
    bars, token = [], None
    while True:
        if token:
            params["page_token"] = token
        resp = requests.get(
            f"https://data.alpaca.markets/v2/stocks/{symbol}/bars",
            params=params,
            headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret},
            timeout=30,
        )
        resp.raise_for_status()
        body = resp.json()
        bars += body.get("bars") or []
        token = body.get("next_page_token")
        if not token:
            break
    df = pd.DataFrame(bars)
    df["t"] = pd.to_datetime(df["t"]).dt.tz_localize(None).dt.normalize()
    df = df.set_index("t").rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})
    return df[["Open", "High", "Low", "Close", "Volume"]]
