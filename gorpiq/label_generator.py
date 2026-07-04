from __future__ import annotations

import numpy as np
import pandas as pd

from gorpiq.config import LABEL_COLUMNS


FORWARD_RETURN_WINDOWS = [2, 3, 5, 10, 15, 20, 30]
FORWARD_WINDOW_DAYS = 30


def calculate_forward_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate forward adjusted-close returns for historical labels.

    Formula: FutureReturn_Xd = AdjClose[t+X] / AdjClose[t] - 1.
    This function intentionally uses future rows because labels are historical
    outcomes for backtesting. These outputs must not be used as current-day
    features or screener inputs.
    """
    result = df.copy()
    adj_close = result["adj_close"].astype(float)

    for window in FORWARD_RETURN_WINDOWS:
        result[f"FutureReturn_{window}d"] = adj_close.shift(-window) / adj_close - 1

    return result


def calculate_forward_window_labels(df: pd.DataFrame, window: int = FORWARD_WINDOW_DAYS) -> pd.DataFrame:
    """Calculate future max/min returns and days-to-event labels.

    The forward window is AdjClose[t+1:t+window], inclusive. Rows without a full
    future window are set to null. This function uses future data by design for
    backtest targets only; its outputs must stay separate from feature inputs.
    """
    result = df.copy()
    closes = result["adj_close"].astype(float).to_numpy()

    max_returns = np.full(len(closes), np.nan)
    min_returns = np.full(len(closes), np.nan)
    days_to_peak = np.full(len(closes), np.nan)
    days_to_drawdown = np.full(len(closes), np.nan)

    for i, current_close in enumerate(closes):
        start = i + 1
        stop = i + window + 1
        if stop > len(closes) or np.isnan(current_close):
            continue

        future_window = closes[start:stop]
        if len(future_window) < window or np.isnan(future_window).any():
            continue

        max_position = int(np.argmax(future_window))
        min_position = int(np.argmin(future_window))
        max_close = future_window[max_position]
        min_close = future_window[min_position]

        max_returns[i] = max_close / current_close - 1
        min_returns[i] = min_close / current_close - 1
        days_to_peak[i] = max_position + 1
        days_to_drawdown[i] = min_position + 1

    result[f"FutureMaxReturn_{window}d"] = max_returns
    result[f"FutureMaxDrawdown_{window}d"] = min_returns
    result[f"FutureDaysToPeak_{window}d"] = days_to_peak
    result[f"FutureDaysToMaxDrawdown_{window}d"] = days_to_drawdown
    return result


def calculate_threshold_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate binary threshold outcomes from 30-day forward window labels.

    Binary labels are null when the required 30-day future max return/drawdown
    is unavailable. They are historical outcomes for backtesting, not features.
    """
    result = df.copy()
    max_return = result["FutureMaxReturn_30d"]
    max_drawdown = result["FutureMaxDrawdown_30d"]

    result["Reached_5pct_Before_30d"] = np.where(max_return.notna(), (max_return >= 0.05).astype(float), np.nan)
    result["Reached_10pct_Before_30d"] = np.where(max_return.notna(), (max_return >= 0.10).astype(float), np.nan)
    result["Reached_15pct_Before_30d"] = np.where(max_return.notna(), (max_return >= 0.15).astype(float), np.nan)

    result["Hit_Negative_5pct_Before_30d"] = np.where(
        max_drawdown.notna(), (max_drawdown <= -0.05).astype(float), np.nan
    )
    result["Hit_Negative_10pct_Before_30d"] = np.where(
        max_drawdown.notna(), (max_drawdown <= -0.10).astype(float), np.nan
    )
    return result


def calculate_labels_for_ticker(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all Build Step 3 labels for one ticker.

    Input rows must include date, ticker, and adj_close. Rows are sorted by date.
    Output columns are forward-looking targets for historical backtesting only;
    do not join them into current-day feature or screener inputs.
    """
    if df.empty:
        return pd.DataFrame(columns=LABEL_COLUMNS)

    result = df.copy()
    result["date"] = pd.to_datetime(result["date"])
    result = result.sort_values("date").reset_index(drop=True)

    result = calculate_forward_returns(result)
    result = calculate_forward_window_labels(result, window=FORWARD_WINDOW_DAYS)
    result = calculate_threshold_labels(result)
    return result.loc[:, LABEL_COLUMNS]


def calculate_labels_for_all_tickers(price_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate forward-looking labels independently for each ticker.

    This uses future adjusted-close rows to create backtest targets. The result
    must remain in `labels_daily` and must not be treated as feature data.
    """
    if price_df.empty:
        return pd.DataFrame(columns=LABEL_COLUMNS)

    required = {"date", "ticker", "adj_close"}
    missing = sorted(required - set(price_df.columns))
    if missing:
        raise ValueError(f"Price data is missing required columns: {missing}")

    frames = []
    for _, ticker_df in price_df.sort_values(["ticker", "date"]).groupby("ticker", sort=True):
        frames.append(calculate_labels_for_ticker(ticker_df))

    if not frames:
        return pd.DataFrame(columns=LABEL_COLUMNS)

    return pd.concat(frames, ignore_index=True).sort_values(["ticker", "date"]).reset_index(drop=True)
