from __future__ import annotations

import math

import pandas as pd

from gorpiq.backtester import (
    calculate_spy_forward_returns,
    load_backtest_dataset,
    run_backtest,
    summarize_backtest_results,
)
from gorpiq.config import FEATURE_COLUMNS, LABEL_COLUMNS, PRICE_COLUMNS
from gorpiq.database import initialize_database, upsert_features, upsert_labels, upsert_prices


def _feature_row(date: str, ticker: str, strength: float) -> dict:
    row = {column: 1.0 for column in FEATURE_COLUMNS}
    row.update(
        {
            "date": pd.Timestamp(date),
            "ticker": ticker,
            "Return_5d": strength,
            "Return_10d": strength,
            "Return_20d": strength,
            "Return_30d": strength,
            "Return_50d": strength,
            "Return_100d": strength,
            "Distance_SMA_20": strength,
            "Distance_SMA_50": strength,
            "Distance_SMA_100": strength,
            "Distance_SMA_200": strength,
            "SMA_10": 110 + strength,
            "SMA_20": 105 + strength,
            "SMA_50": 100,
            "SMA_100": 95,
            "SMA_200": 90,
            "RSI_5": 60,
            "RSI_14": 60,
            "RSI_21": 60,
            "ATR_Pct_14": max(0.01, 0.08 - strength),
            "ATR_Pct_20": max(0.01, 0.08 - strength),
        }
    )
    return row


def _label_row(date: str, ticker: str, future_return: float | None) -> dict:
    row = {column: 0.0 for column in LABEL_COLUMNS}
    row.update(
        {
            "date": pd.Timestamp(date),
            "ticker": ticker,
            "FutureReturn_5d": future_return,
            "FutureMaxReturn_30d": 0.12,
            "FutureMaxDrawdown_30d": -0.04,
            "Reached_5pct_Before_30d": 1.0,
            "Reached_10pct_Before_30d": 1.0,
            "Hit_Negative_5pct_Before_30d": 0.0,
        }
    )
    return row


def _spy_prices() -> pd.DataFrame:
    closes = [100, 101, 102, 103, 104, 110, 111]
    dates = pd.date_range("2024-01-02", periods=len(closes), freq="B")
    rows = []
    for date, close in zip(dates, closes):
        rows.append(
            {
                "date": date,
                "ticker": "SPY",
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "adj_close": close,
                "volume": 1000,
                "data_source": "synthetic",
            }
        )
    return pd.DataFrame(rows).loc[:, PRICE_COLUMNS]


def test_backtester_selects_top_n_and_calculates_excess_return() -> None:
    features = pd.DataFrame(
        [
            _feature_row("2024-01-02", "AAA", 0.20),
            _feature_row("2024-01-02", "BBB", 0.05),
            _feature_row("2024-01-03", "AAA", 0.02),
            _feature_row("2024-01-03", "BBB", 0.15),
        ]
    )
    labels = pd.DataFrame(
        [
            _label_row("2024-01-02", "AAA", 0.20),
            _label_row("2024-01-02", "BBB", 0.05),
            _label_row("2024-01-03", "AAA", 0.01),
            _label_row("2024-01-03", "BBB", 0.10),
        ]
    )
    spy_returns = calculate_spy_forward_returns(_spy_prices(), holding_period=5)

    results = run_backtest(features, labels, spy_returns, holding_period=5, top_n=1)

    assert results["ticker"].tolist() == ["AAA", "BBB"]
    assert math.isclose(results.iloc[0]["SPY_FutureReturn"], 110 / 100 - 1)
    assert math.isclose(results.iloc[0]["ExcessReturnVsSPY"], 0.20 - (110 / 100 - 1))


def test_backtester_excludes_rows_with_null_selected_labels() -> None:
    features = pd.DataFrame([_feature_row("2024-01-02", "AAA", 0.20), _feature_row("2024-01-02", "BBB", 0.05)])
    labels = pd.DataFrame([_label_row("2024-01-02", "AAA", None), _label_row("2024-01-02", "BBB", 0.05)])
    spy_returns = calculate_spy_forward_returns(_spy_prices(), holding_period=5)

    results = run_backtest(features, labels, spy_returns, holding_period=5, top_n=2)

    assert results["ticker"].tolist() == ["BBB"]


def test_spy_is_excluded_from_candidate_ranking_by_default(tmp_path) -> None:
    db_path = tmp_path / "backtest.sqlite"
    initialize_database(db_path)
    features = pd.DataFrame([_feature_row("2024-01-02", "AAA", 0.10), _feature_row("2024-01-02", "SPY", 0.30)])
    labels = pd.DataFrame([_label_row("2024-01-02", "AAA", 0.10), _label_row("2024-01-02", "SPY", 0.20)])
    upsert_features(features, db_path=db_path)
    upsert_labels(labels, db_path=db_path)
    upsert_prices(_spy_prices(), db_path=db_path)

    loaded_features, _, _, validation = load_backtest_dataset(
        ["AAA", "SPY"], "2024-01-02", "2024-01-02", holding_period=5, db_path=db_path
    )

    assert validation.is_valid
    assert loaded_features["ticker"].tolist() == ["AAA"]


def test_summary_metrics_are_calculated() -> None:
    results = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "ticker": ["AAA", "BBB"],
            "SelectedFutureReturn": [0.20, -0.05],
            "SPY_FutureReturn": [0.10, 0.00],
            "ExcessReturnVsSPY": [0.10, -0.05],
            "FutureMaxReturn_30d": [0.25, 0.03],
            "FutureMaxDrawdown_30d": [-0.04, -0.08],
            "Reached_5pct_Before_30d": [1.0, 0.0],
            "Reached_10pct_Before_30d": [1.0, 0.0],
            "Hit_Negative_5pct_Before_30d": [0.0, 1.0],
        }
    )

    summary = summarize_backtest_results(results)

    assert summary["Number of backtest dates"] == 2
    assert summary["Number of selected trades"] == 2
    assert math.isclose(summary["Average selected future return"], 0.075)
    assert math.isclose(summary["Percent of trades beating SPY"], 0.5)
