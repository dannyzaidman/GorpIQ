from __future__ import annotations

import numpy as np
import pandas as pd

from gorpiq.config import FEATURE_COLUMNS


RETURN_WINDOWS = [1, 2, 3, 5, 10, 15, 20, 30, 50, 100, 200]
SMA_WINDOWS = [10, 20, 50, 100, 200]
RSI_WINDOWS = [5, 14, 21]
ATR_WINDOWS = [14, 20]


def calculate_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate trailing adjusted-close returns without using future rows.

    Return_Xd = AdjClose[t] / AdjClose[t-X] - 1. Return_YTD uses the first
    available trading day in the same calendar year for that ticker.
    """
    result = df.copy()
    adj_close = result["adj_close"].astype(float)

    for window in RETURN_WINDOWS:
        result[f"Return_{window}d"] = adj_close / adj_close.shift(window) - 1

    years = pd.to_datetime(result["date"]).dt.year
    year_start = adj_close.groupby(years).transform("first")
    result["Return_YTD"] = adj_close / year_start - 1
    return result


def calculate_smas(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate trailing simple moving averages from adjusted close.

    SMA_X is the average adjusted close over the trailing X rows, including
    the current date. Values are null until X rows are available.
    """
    result = df.copy()
    adj_close = result["adj_close"].astype(float)

    for window in SMA_WINDOWS:
        result[f"SMA_{window}"] = adj_close.rolling(window=window, min_periods=window).mean()
    return result


def calculate_sma_distances(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate adjusted-close distance from each trailing SMA.

    Distance_SMA_X = AdjClose[t] / SMA_X[t] - 1. Distances stay null while the
    corresponding SMA is null.
    """
    result = df.copy()
    adj_close = result["adj_close"].astype(float)

    for window in SMA_WINDOWS:
        sma = result[f"SMA_{window}"].astype(float)
        result[f"Distance_SMA_{window}"] = adj_close / sma - 1
    return result


def calculate_rsi(df: pd.DataFrame, window: int) -> pd.Series:
    """Calculate Wilder RSI from adjusted close.

    The initial average gain/loss is the simple average of the first `window`
    price changes. Later rows use Wilder smoothing:
    avg = (prior_avg * (window - 1) + current_value) / window.
    """
    closes = df["adj_close"].astype(float).to_numpy()
    deltas = np.diff(closes, prepend=np.nan)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = np.full(len(closes), np.nan)
    avg_loss = np.full(len(closes), np.nan)
    rsi = np.full(len(closes), np.nan)

    if len(closes) <= window:
        return pd.Series(rsi, index=df.index, name=f"RSI_{window}")

    avg_gain[window] = np.nanmean(gains[1 : window + 1])
    avg_loss[window] = np.nanmean(losses[1 : window + 1])

    for i in range(window + 1, len(closes)):
        avg_gain[i] = ((avg_gain[i - 1] * (window - 1)) + gains[i]) / window
        avg_loss[i] = ((avg_loss[i - 1] * (window - 1)) + losses[i]) / window

    for i in range(window, len(closes)):
        if avg_loss[i] == 0:
            rsi[i] = 100.0 if avg_gain[i] > 0 else 50.0
        else:
            relative_strength = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + relative_strength))

    return pd.Series(rsi, index=df.index, name=f"RSI_{window}")


def calculate_atr(df: pd.DataFrame, window: int) -> pd.Series:
    """Calculate average true range from stored unadjusted OHLC rows.

    True Range is max(high-low, abs(high-prior close), abs(low-prior close)).
    ATR_X is the trailing rolling average of true range over X rows, including
    the current date. The app stores adjusted close but not adjusted OHLC, so
    true range uses stored unadjusted high, low, and close.
    """
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    prior_close = df["close"].astype(float).shift(1)

    true_range = pd.concat(
        [
            high - low,
            (high - prior_close).abs(),
            (low - prior_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return true_range.rolling(window=window, min_periods=window).mean().rename(f"ATR_{window}")


def calculate_features_for_ticker(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate the Build Step 2 feature subset for one ticker.

    Input rows must include date, ticker, high, low, close, and adj_close.
    Calculations use only the current row and earlier rows after sorting by date.
    """
    if df.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS)

    result = df.copy()
    result["date"] = pd.to_datetime(result["date"])
    result = result.sort_values("date").reset_index(drop=True)

    result = calculate_returns(result)
    result = calculate_smas(result)
    result = calculate_sma_distances(result)

    for window in RSI_WINDOWS:
        result[f"RSI_{window}"] = calculate_rsi(result, window)

    for window in ATR_WINDOWS:
        result[f"ATR_{window}"] = calculate_atr(result, window)
        result[f"ATR_Pct_{window}"] = result[f"ATR_{window}"] / result["adj_close"].astype(float)

    return result.loc[:, FEATURE_COLUMNS]


def calculate_features_for_all_tickers(price_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate Build Step 2 features independently for each ticker."""
    if price_df.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS)

    required = {"date", "ticker", "high", "low", "close", "adj_close"}
    missing = sorted(required - set(price_df.columns))
    if missing:
        raise ValueError(f"Price data is missing required columns: {missing}")

    frames = []
    for _, ticker_df in price_df.sort_values(["ticker", "date"]).groupby("ticker", sort=True):
        frames.append(calculate_features_for_ticker(ticker_df))

    if not frames:
        return pd.DataFrame(columns=FEATURE_COLUMNS)

    return pd.concat(frames, ignore_index=True).sort_values(["ticker", "date"]).reset_index(drop=True)
