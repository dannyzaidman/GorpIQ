from __future__ import annotations

import math

import pandas as pd

from gorpiq.database import initialize_database, load_features, upsert_features
from gorpiq.feature_engineering import (
    calculate_atr,
    calculate_features_for_all_tickers,
    calculate_features_for_ticker,
    calculate_rsi,
)


def _synthetic_prices(days: int = 25, ticker: str = "TEST") -> pd.DataFrame:
    closes = [float(i) for i in range(1, days + 1)]
    dates = pd.date_range("2024-01-02", periods=days, freq="B")
    return pd.DataFrame(
        {
            "date": dates,
            "ticker": ticker,
            "open": closes,
            "high": [close + 1 for close in closes],
            "low": [close - 1 for close in closes],
            "close": closes,
            "adj_close": closes,
            "volume": [1000] * days,
            "data_source": ["synthetic"] * days,
        }
    )


def test_return_calculations() -> None:
    features = calculate_features_for_ticker(_synthetic_prices())

    row = features.iloc[5]
    assert row["Return_1d"] == 6 / 5 - 1
    assert row["Return_5d"] == 6 / 1 - 1
    assert pd.isna(features.loc[3, "Return_5d"])


def test_sma_calculations() -> None:
    features = calculate_features_for_ticker(_synthetic_prices())

    assert features.loc[9, "SMA_10"] == 5.5
    assert features.loc[19, "SMA_20"] == 10.5
    assert pd.isna(features.loc[8, "SMA_10"])
    assert pd.isna(features.loc[18, "SMA_20"])


def test_distance_sma_calculation() -> None:
    features = calculate_features_for_ticker(_synthetic_prices())

    assert math.isclose(features.loc[9, "Distance_SMA_10"], 10 / 5.5 - 1)
    assert pd.isna(features.loc[8, "Distance_SMA_10"])


def test_rsi_wilder_all_gain_series() -> None:
    prices = _synthetic_prices(days=8)
    rsi = calculate_rsi(prices, window=5)

    assert rsi.iloc[:5].isna().all()
    assert rsi.iloc[5] == 100.0
    assert rsi.iloc[7] == 100.0


def test_atr_and_atr_pct_calculations() -> None:
    prices = _synthetic_prices(days=20)
    atr = calculate_atr(prices, window=14)
    features = calculate_features_for_ticker(prices)

    assert pd.isna(atr.iloc[12])
    assert atr.iloc[13] == 2.0
    assert features.loc[13, "ATR_14"] == 2.0
    assert math.isclose(features.loc[13, "ATR_Pct_14"], 2 / 14)


def test_features_are_calculated_independently_by_ticker() -> None:
    test_prices = _synthetic_prices(days=12, ticker="TEST")
    other_prices = _synthetic_prices(days=12, ticker="OTHER")
    other_prices["adj_close"] = other_prices["adj_close"] * 10
    other_prices["close"] = other_prices["close"] * 10
    other_prices["high"] = other_prices["high"] * 10
    other_prices["low"] = other_prices["low"] * 10

    features = calculate_features_for_all_tickers(pd.concat([test_prices, other_prices], ignore_index=True))

    test_row = features[(features["ticker"] == "TEST") & (features["date"] == pd.Timestamp("2024-01-15"))].iloc[0]
    other_row = features[(features["ticker"] == "OTHER") & (features["date"] == pd.Timestamp("2024-01-15"))].iloc[0]
    assert test_row["SMA_10"] == 5.5
    assert other_row["SMA_10"] == 55.0


def test_feature_upsert_and_load_roundtrip(tmp_path) -> None:
    db_path = tmp_path / "features.sqlite"
    initialize_database(db_path)
    features = calculate_features_for_ticker(_synthetic_prices(days=12))

    assert upsert_features(features, db_path=db_path) == 12
    assert upsert_features(features, db_path=db_path) == 12

    loaded = load_features(tickers=["TEST"], start_date="2024-01-02", end_date="2024-01-31", db_path=db_path)
    assert len(loaded) == 12
    assert loaded["ticker"].unique().tolist() == ["TEST"]
