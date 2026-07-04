from __future__ import annotations

import math

import pandas as pd

from gorpiq.scoring import (
    percentile_rank_series,
    score_candidates,
    score_mean_reversion,
    score_momentum,
    score_trend,
    score_volatility_risk,
)


def _features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02"] * 3),
            "ticker": ["AAA", "BBB", "CCC"],
            "Return_5d": [0.10, 0.05, -0.01],
            "Return_10d": [0.12, 0.04, -0.02],
            "Return_20d": [0.20, 0.03, -0.03],
            "Return_30d": [0.25, 0.02, -0.04],
            "Return_50d": [0.30, 0.01, -0.05],
            "Return_100d": [0.35, 0.00, -0.06],
            "Distance_SMA_20": [0.05, 0.01, -0.02],
            "Distance_SMA_50": [0.06, -0.01, -0.03],
            "Distance_SMA_100": [0.07, -0.02, -0.04],
            "Distance_SMA_200": [0.08, -0.03, -0.05],
            "SMA_10": [110, 100, 90],
            "SMA_20": [108, 101, 91],
            "SMA_50": [104, 102, 92],
            "SMA_100": [100, 103, 93],
            "SMA_200": [95, 104, 94],
            "RSI_5": [65, 75, 85],
            "RSI_14": [60, 78, 88],
            "RSI_21": [55, 72, 82],
            "ATR_Pct_14": [0.02, 0.05, 0.10],
            "ATR_Pct_20": [0.025, 0.055, 0.11],
        }
    )


def test_percentile_ranking_behavior() -> None:
    ranks = percentile_rank_series(pd.Series([1, 2, 3]))
    inverted = percentile_rank_series(pd.Series([1, 2, 3]), higher_is_better=False)

    assert math.isclose(ranks.iloc[0], 100 / 3)
    assert ranks.iloc[2] == 100
    assert inverted.iloc[0] == 100
    assert math.isclose(inverted.iloc[2], 100 / 3)


def test_momentum_scoring_behavior() -> None:
    scores = score_momentum(_features())
    assert scores.iloc[0] > scores.iloc[1] > scores.iloc[2]


def test_trend_scoring_behavior() -> None:
    scores = score_trend(_features())
    assert scores.iloc[0] > scores.iloc[1] > scores.iloc[2]


def test_rsi_overextension_penalty_behavior() -> None:
    scores = score_mean_reversion(_features())
    assert scores.iloc[0] > scores.iloc[1] > scores.iloc[2]


def test_atr_risk_penalty_behavior() -> None:
    scores = score_volatility_risk(_features())
    assert scores.iloc[0] > scores.iloc[1] > scores.iloc[2]


def test_total_score_calculation() -> None:
    scored = score_candidates(_features())
    row = scored[scored["ticker"] == "AAA"].iloc[0]
    expected = (
        row["MomentumScore"] * 0.35
        + row["TrendScore"] * 0.35
        + row["MeanReversionScore"] * 0.15
        + row["VolatilityRiskScore"] * 0.15
    )
    assert math.isclose(row["TotalScore"], expected)
    assert row["TotalScore"] > scored[scored["ticker"] == "CCC"].iloc[0]["TotalScore"]
