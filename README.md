# stocks-bot

Backtesting-first US stock trading research bot. Paper only: nothing here places orders.

- `stocksbot/data.py`: CSV loader, Alpaca market data (your own keys via env), Finviz Elite export reader.
- `stocksbot/strategies.py`: SMA cross, Donchian breakout, RSI(2) reversion, Bollinger reversion, 12-1 momentum, ensemble vote.
- `stocksbot/backtest.py`: daily engine, next-bar execution, fees + slippage, vol targeting, drawdown kill switch.
- `stocksbot/walkforward.py`: rolling 4y-train / 1y-test parameter selection; reports out-of-sample only.
- `stocksbot/webhook.py`: receiver for your own TradingView Pine Script alerts.
- `run_backtest.py`: runs everything and writes `results/`.

## Run
    pip install -r requirements.txt
    python -m pytest -q tests
    python run_backtest.py

## First results (sample data, 1995-2016)
Ensemble, walk-forward out of sample, median of 8 tickers: ~3% CAGR, ~12% max drawdown,
vs buy-and-hold ~16% CAGR, ~71% max drawdown. Sample data is old and survivor-biased;
use Alpaca for current data.
