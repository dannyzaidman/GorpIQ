# GorpIQ

GorpIQ is a local Streamlit app for personal stock trading research and decision support. The current build supports watchlist entry, daily OHLCV download through yfinance, local SQLite storage, inspection of downloaded data, and an initial feature-engineering slice.

This is not financial advice, not an auto-trading system, and not a broker integration. It does not place trades. Past performance does not guarantee future results.

## What The Current Build Does

- Accepts manually entered tickers.
- Accepts a CSV watchlist upload.
- Downloads daily OHLCV data from Yahoo Finance through yfinance.
- Stores normalized daily price rows in local SQLite.
- Shows stored ticker coverage and a preview of downloaded OHLCV rows.
- Calculates and stores initial MVP features:
  - absolute momentum returns
  - simple moving averages
  - moving average distance features
  - RSI
  - ATR and ATR percent
- Shows stored feature coverage, feature previews, missing-value counts, and CSV export.
- Provides shell pages for the planned label, backtest, screener, trade tracker, and sell alert modules.

## What It Does Not Do Yet

- Generate forward-looking labels.
- Run backtests.
- Rank current candidates.
- Log trades.
- Generate sell alerts.
- Connect to a broker or place trades.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

The app stores its SQLite database at:

```text
data/gorpiq.sqlite
```

## Enter A Watchlist

Open the `Watchlist & Data Download` page. Enter tickers separated by commas, spaces, or new lines. You can also upload a CSV. If the CSV has a `ticker` column, that column is used. Otherwise, the first column is treated as tickers.

## Download Data

Choose a start date and end date, then select `Download and store daily prices`. The app downloads adjusted and unadjusted daily price fields from yfinance and upserts them into SQLite.

## Key Limitations

- Yahoo/yfinance data may be incomplete, delayed, unavailable, or adjusted differently than professional datasets.
- Using today's stock universe for historical testing can introduce survivorship bias.
- Later backtests must avoid using future data in features.
- Forward-looking labels are for historical testing only and must never be used as current-day features.

## Next Build Step

Add the forward-looking label generator for historical backtesting. Labels should be stored separately from features and must never be used as current-day inputs.
