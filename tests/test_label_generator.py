from __future__ import annotations

import math

import pandas as pd

from gorpiq.database import initialize_database, load_features, load_labels, upsert_labels
from gorpiq.label_generator import calculate_labels_for_all_tickers, calculate_labels_for_ticker


def _synthetic_prices(ticker: str = "TEST") -> pd.DataFrame:
    closes = [
        100.0,
        102.0,
        106.0,
        108.0,
        94.0,
        115.0,
        89.0,
        116.0,
        101.0,
        102.0,
        103.0,
        104.0,
        105.0,
        106.0,
        107.0,
        108.0,
        109.0,
        110.0,
        111.0,
        112.0,
        113.0,
        114.0,
        105.0,
        106.0,
        107.0,
        108.0,
        109.0,
        110.0,
        111.0,
        112.0,
        112.0,
        113.0,
        114.0,
        115.0,
        116.0,
        117.0,
    ]
    dates = pd.date_range("2024-01-02", periods=len(closes), freq="B")
    return pd.DataFrame(
        {
            "date": dates,
            "ticker": ticker,
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "adj_close": closes,
            "volume": [1000] * len(closes),
            "data_source": ["synthetic"] * len(closes),
        }
    )


def test_forward_return_labels() -> None:
    labels = calculate_labels_for_ticker(_synthetic_prices())
    row = labels.iloc[0]

    assert math.isclose(row["FutureReturn_2d"], 106 / 100 - 1)
    assert math.isclose(row["FutureReturn_5d"], 115 / 100 - 1)
    assert math.isclose(row["FutureReturn_30d"], 112 / 100 - 1)


def test_forward_window_max_min_and_days() -> None:
    labels = calculate_labels_for_ticker(_synthetic_prices())
    row = labels.iloc[0]

    assert math.isclose(row["FutureMaxReturn_30d"], 116 / 100 - 1)
    assert math.isclose(row["FutureMaxDrawdown_30d"], 89 / 100 - 1)
    assert row["FutureDaysToPeak_30d"] == 7
    assert row["FutureDaysToMaxDrawdown_30d"] == 6


def test_threshold_labels() -> None:
    labels = calculate_labels_for_ticker(_synthetic_prices())
    row = labels.iloc[0]

    assert row["Reached_5pct_Before_30d"] == 1.0
    assert row["Reached_10pct_Before_30d"] == 1.0
    assert row["Reached_15pct_Before_30d"] == 1.0
    assert row["Hit_Negative_5pct_Before_30d"] == 1.0
    assert row["Hit_Negative_10pct_Before_30d"] == 1.0


def test_rows_without_enough_future_data_have_null_labels() -> None:
    labels = calculate_labels_for_ticker(_synthetic_prices())
    last_row = labels.iloc[-1]

    assert pd.isna(last_row["FutureReturn_2d"])
    assert pd.isna(last_row["FutureReturn_30d"])
    assert pd.isna(last_row["FutureMaxReturn_30d"])
    assert pd.isna(last_row["FutureMaxDrawdown_30d"])
    assert pd.isna(last_row["FutureDaysToPeak_30d"])
    assert pd.isna(last_row["Reached_5pct_Before_30d"])
    assert pd.isna(last_row["Hit_Negative_10pct_Before_30d"])


def test_labels_are_calculated_independently_by_ticker() -> None:
    test_prices = _synthetic_prices("TEST")
    other_prices = _synthetic_prices("OTHER")
    other_prices["adj_close"] = other_prices["adj_close"] * 2

    labels = calculate_labels_for_all_tickers(pd.concat([test_prices, other_prices], ignore_index=True))

    test_row = labels[(labels["ticker"] == "TEST") & (labels["date"] == pd.Timestamp("2024-01-02"))].iloc[0]
    other_row = labels[(labels["ticker"] == "OTHER") & (labels["date"] == pd.Timestamp("2024-01-02"))].iloc[0]
    assert math.isclose(test_row["FutureReturn_2d"], other_row["FutureReturn_2d"])


def test_label_upsert_roundtrip_and_no_duplicates(tmp_path) -> None:
    db_path = tmp_path / "labels.sqlite"
    initialize_database(db_path)
    labels = calculate_labels_for_ticker(_synthetic_prices())

    assert upsert_labels(labels, db_path=db_path) == len(labels)
    assert upsert_labels(labels, db_path=db_path) == len(labels)

    loaded = load_labels(tickers=["TEST"], start_date="2024-01-02", end_date="2024-03-31", db_path=db_path)
    assert len(loaded) == len(labels)
    assert loaded["ticker"].unique().tolist() == ["TEST"]


def test_labels_are_stored_separately_from_features(tmp_path) -> None:
    db_path = tmp_path / "separate.sqlite"
    initialize_database(db_path)
    labels = calculate_labels_for_ticker(_synthetic_prices())

    upsert_labels(labels, db_path=db_path)

    loaded_labels = load_labels(tickers=["TEST"], db_path=db_path)
    loaded_features = load_features(tickers=["TEST"], db_path=db_path)
    assert not loaded_labels.empty
    assert loaded_features.empty
