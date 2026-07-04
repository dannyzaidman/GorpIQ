# GorpIQ

GorpIQ is a local Streamlit app for personal stock trading research and decision support. The current build supports watchlist entry, daily OHLCV download through yfinance, local SQLite storage, inspection of downloaded data, initial feature engineering, and forward-looking label generation for historical backtesting.

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
- Calculates and stores forward-looking labels:
  - 2, 3, 5, 10, 15, 20, and 30 trading day future returns
  - 30-day future max return and max drawdown
  - days to future peak and max drawdown
  - +5%, +10%, +15%, -5%, and -10% threshold outcomes
- Shows stored label coverage, label previews, missing-value counts, and CSV export.
- Provides shell pages for the planned backtest, screener, trade tracker, and sell alert modules.

## What It Does Not Do Yet

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

## Generate Labels

Open the `Label Generation` page after downloading price history. Select tickers and a label date range, then select `Generate forward-looking labels`.

Labels intentionally use future adjusted close data. They are stored in `labels_daily` as historical outcomes for backtesting only. Do not use label columns as current-day features, screener inputs, or trading signals.

## Key Limitations

- Yahoo/yfinance data may be incomplete, delayed, unavailable, or adjusted differently than professional datasets.
- Using today's stock universe for historical testing can introduce survivorship bias.
- Later backtests must avoid using future data in features.
- Forward-looking labels are for historical testing only and must never be used as current-day features.

## Next Build Step

Add a simple backtesting page that uses historical features, scores/rules, and stored labels to evaluate candidate selection against future outcomes.
