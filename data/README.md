# Sample data sources

All public, downloaded from GitHub raw files (no scraping of any site):

| File | Contents | Source |
|---|---|---|
| nvda-1999-2014.csv, orcl-1995-2014.csv, yhoo-1996-2014.csv | Daily OHLCV + Adj Close | github.com/mementum/backtrader `datas/` |
| stockdata-2007-2016.csv | Daily adjusted closes: MSFT, IBM, SBUX, AAPL, S&P 500 index | github.com/plotly/datasets |
| sp500-monthly-1871.csv | Monthly S&P 500 (Shiller) to 2026 | github.com/datasets/s-and-p-500 (not used yet) |

These end in 2014-2016 and the tickers are well-known survivors, so results lean optimistic.
For current data use `stocksbot.data.fetch_alpaca_daily` with your own free Alpaca keys.
