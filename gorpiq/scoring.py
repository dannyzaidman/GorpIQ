from __future__ import annotations

import numpy as np
import pandas as pd


MOMENTUM_COLUMNS = ["Return_5d", "Return_10d", "Return_20d", "Return_30d", "Return_50d", "Return_100d"]
DISTANCE_COLUMNS = ["Distance_SMA_20", "Distance_SMA_50", "Distance_SMA_100", "Distance_SMA_200"]
SMA_COLUMNS = ["SMA_10", "SMA_20", "SMA_50", "SMA_100", "SMA_200"]
RSI_COLUMNS = ["RSI_5", "RSI_14", "RSI_21"]
ATR_PCT_COLUMNS = ["ATR_Pct_14", "ATR_Pct_20"]


def percentile_rank_series(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """Return percentile ranks on a 0-100 scale, preserving nulls."""
    numeric = pd.to_numeric(series, errors="coerce")
    ranked = numeric.rank(method="average", pct=True, ascending=higher_is_better) * 100
    return ranked.where(numeric.notna())


def _available_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if column in df.columns]


def score_momentum(features_df: pd.DataFrame) -> pd.Series:
    """Score momentum using same-date cross-sectional return percentiles."""
    columns = _available_columns(features_df, MOMENTUM_COLUMNS)
    if not columns:
        return pd.Series(np.nan, index=features_df.index)

    scores = [percentile_rank_series(features_df[column], higher_is_better=True) for column in columns]
    return pd.concat(scores, axis=1).mean(axis=1, skipna=True).clip(0, 100)


def score_trend(features_df: pd.DataFrame) -> pd.Series:
    """Score moving-average trend from price-vs-SMA distance and SMA alignment."""
    result = pd.Series(0.0, index=features_df.index)
    components = 0

    for column in _available_columns(features_df, DISTANCE_COLUMNS):
        result += (pd.to_numeric(features_df[column], errors="coerce") > 0).astype(float) * 100
        components += 1

    if set(SMA_COLUMNS).issubset(features_df.columns):
        sma = features_df[SMA_COLUMNS].apply(pd.to_numeric, errors="coerce")
        result += (sma["SMA_20"] > sma["SMA_50"]).astype(float) * 100
        result += (sma["SMA_50"] > sma["SMA_100"]).astype(float) * 100
        result += (sma["SMA_100"] > sma["SMA_200"]).astype(float) * 100
        components += 3

    if components == 0:
        return pd.Series(np.nan, index=features_df.index)
    return (result / components).clip(0, 100)


def _score_single_rsi(rsi: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(rsi, errors="coerce")
    score = pd.Series(np.nan, index=rsi.index)
    score[(numeric >= 30) & (numeric <= 70)] = 100
    score[(numeric > 70) & (numeric <= 80)] = 60
    score[numeric > 80] = 20
    score[(numeric >= 20) & (numeric < 30)] = 70
    score[numeric < 20] = 40
    return score


def score_mean_reversion(features_df: pd.DataFrame) -> pd.Series:
    """Score RSI overextension, penalizing severely overbought readings."""
    columns = _available_columns(features_df, RSI_COLUMNS)
    if not columns:
        return pd.Series(np.nan, index=features_df.index)

    scores = [_score_single_rsi(features_df[column]) for column in columns]
    return pd.concat(scores, axis=1).mean(axis=1, skipna=True).clip(0, 100)


def score_volatility_risk(features_df: pd.DataFrame) -> pd.Series:
    """Score ATR percent risk using inverted cross-sectional percentiles."""
    columns = _available_columns(features_df, ATR_PCT_COLUMNS)
    if not columns:
        return pd.Series(np.nan, index=features_df.index)

    scores = [percentile_rank_series(features_df[column], higher_is_better=False) for column in columns]
    return pd.concat(scores, axis=1).mean(axis=1, skipna=True).clip(0, 100)


def generate_reason_codes(row: pd.Series) -> str:
    """Generate compact explanation text for a scored candidate row."""
    reasons: list[str] = []
    if row.get("Return_20d", 0) > 0 and row.get("Return_50d", 0) > 0:
        reasons.append("Strong 20-day and 50-day momentum")
    elif row.get("MomentumScore", 0) < 40:
        reasons.append("Weak momentum")

    if all(pd.notna(row.get(column)) and row.get(column) > 0 for column in ["Distance_SMA_20", "Distance_SMA_50"]):
        reasons.append("Price above key moving averages")
    if pd.notna(row.get("Distance_SMA_200")) and row.get("Distance_SMA_200") < 0:
        reasons.append("Below long-term moving average")

    if pd.notna(row.get("RSI_14")) and row.get("RSI_14") <= 70:
        reasons.append("RSI not severely overextended")
    elif pd.notna(row.get("RSI_14")):
        reasons.append("RSI overextended")

    if pd.notna(row.get("ATR_Pct_14")) and row.get("ATR_Pct_14") <= 0.06:
        reasons.append("ATR risk acceptable")
    elif pd.notna(row.get("ATR_Pct_14")):
        reasons.append("High volatility")

    return "; ".join(reasons) if reasons else "Mixed signal profile"


def score_candidates(features_df: pd.DataFrame) -> pd.DataFrame:
    """Score candidates from feature columns only.

    This function intentionally ignores labels. Labels are future outcomes and
    must not be used as scoring inputs.
    """
    if features_df.empty:
        return features_df.copy()

    scored_frames = []
    for _, date_df in features_df.groupby("date", sort=True):
        scored = date_df.copy()
        scored["MomentumScore"] = score_momentum(scored)
        scored["TrendScore"] = score_trend(scored)
        scored["MeanReversionScore"] = score_mean_reversion(scored)
        scored["VolatilityRiskScore"] = score_volatility_risk(scored)
        scored["TotalScore"] = (
            scored["MomentumScore"].fillna(0) * 0.35
            + scored["TrendScore"].fillna(0) * 0.35
            + scored["MeanReversionScore"].fillna(0) * 0.15
            + scored["VolatilityRiskScore"].fillna(0) * 0.15
        ).clip(0, 100)
        scored["ReasonCodes"] = scored.apply(generate_reason_codes, axis=1)
        scored_frames.append(scored)

    return pd.concat(scored_frames, ignore_index=True)
