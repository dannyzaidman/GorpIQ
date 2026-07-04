from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from gorpiq.database import load_features, load_labels, load_prices
from gorpiq.scoring import score_candidates


HOLDING_PERIOD_LABELS = {
    2: "FutureReturn_2d",
    3: "FutureReturn_3d",
    5: "FutureReturn_5d",
    10: "FutureReturn_10d",
    15: "FutureReturn_15d",
    20: "FutureReturn_20d",
    30: "FutureReturn_30d",
}

SUPPORTING_LABEL_COLUMNS = [
    "FutureMaxReturn_30d",
    "FutureMaxDrawdown_30d",
    "Reached_5pct_Before_30d",
    "Reached_10pct_Before_30d",
    "Reached_15pct_Before_30d",
    "Hit_Negative_5pct_Before_30d",
    "Hit_Negative_10pct_Before_30d",
]


@dataclass
class BacktestValidation:
    is_valid: bool
    messages: list[str]


def calculate_spy_forward_returns(spy_prices: pd.DataFrame, holding_period: int) -> pd.DataFrame:
    """Calculate SPY benchmark forward returns from adjusted close prices."""
    if spy_prices.empty:
        return pd.DataFrame(columns=["date", "SPY_FutureReturn"])

    prices = spy_prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices[prices["ticker"].str.upper() == "SPY"].sort_values("date").reset_index(drop=True)
    prices["SPY_FutureReturn"] = prices["adj_close"].astype(float).shift(-holding_period) / prices["adj_close"].astype(float) - 1
    return prices.loc[:, ["date", "SPY_FutureReturn"]]


def validate_backtest_inputs(
    features_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    spy_returns_df: pd.DataFrame,
    holding_period: int,
) -> BacktestValidation:
    messages: list[str] = []
    label_column = HOLDING_PERIOD_LABELS.get(holding_period)

    if label_column is None:
        messages.append(f"Unsupported holding period: {holding_period}.")
    if features_df.empty:
        messages.append("Candidate feature data is missing for the selected tickers/date range.")
    if labels_df.empty:
        messages.append("Forward-looking labels are missing for the selected tickers/date range.")
    elif label_column and label_column not in labels_df.columns:
        messages.append(f"Required label column is missing: {label_column}.")
    if spy_returns_df.empty or spy_returns_df["SPY_FutureReturn"].dropna().empty:
        messages.append("SPY benchmark data is required for backtesting. Please download SPY data for this date range first.")

    return BacktestValidation(is_valid=not messages, messages=messages)


def load_backtest_dataset(
    tickers: Iterable[str],
    start_date: date | str,
    end_date: date | str,
    holding_period: int,
    include_spy_as_candidate: bool = False,
    db_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, BacktestValidation]:
    """Load features, labels, and SPY benchmark returns for a backtest."""
    candidate_tickers = [ticker.upper().strip() for ticker in tickers if ticker and ticker.strip()]
    if not include_spy_as_candidate:
        candidate_tickers = [ticker for ticker in candidate_tickers if ticker != "SPY"]

    load_kwargs = {"db_path": db_path} if db_path is not None else {}
    features = load_features(tickers=candidate_tickers, start_date=start_date, end_date=end_date, **load_kwargs)
    labels = load_labels(tickers=candidate_tickers, start_date=start_date, end_date=end_date, **load_kwargs)
    spy_prices = load_prices(tickers=["SPY"], **load_kwargs)
    spy_returns = calculate_spy_forward_returns(spy_prices, holding_period)
    if not spy_returns.empty:
        spy_returns = spy_returns[
            (spy_returns["date"].dt.date >= pd.to_datetime(start_date).date())
            & (spy_returns["date"].dt.date <= pd.to_datetime(end_date).date())
        ].reset_index(drop=True)
    validation = validate_backtest_inputs(features, labels, spy_returns, holding_period)
    return features, labels, spy_returns, validation


def run_backtest(
    features_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    spy_returns_df: pd.DataFrame,
    holding_period: int,
    top_n: int = 3,
    minimum_score: float = 0,
) -> pd.DataFrame:
    """Run an initial historical ranking backtest.

    Features are used for same-date scoring. Labels are joined only after
    candidate selection to evaluate future outcomes.
    """
    label_column = HOLDING_PERIOD_LABELS[holding_period]
    if features_df.empty or labels_df.empty or spy_returns_df.empty:
        return pd.DataFrame()

    scored = score_candidates(features_df)
    scored = scored[scored["TotalScore"] >= minimum_score].copy()
    if scored.empty:
        return pd.DataFrame()

    selected = (
        scored.sort_values(["date", "TotalScore", "ticker"], ascending=[True, False, True])
        .groupby("date", group_keys=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    label_columns = ["date", "ticker", label_column, *SUPPORTING_LABEL_COLUMNS]
    labels = labels_df.loc[:, [column for column in label_columns if column in labels_df.columns]].copy()
    results = selected.merge(labels, on=["date", "ticker"], how="left")
    results = results[results[label_column].notna()].copy()
    if results.empty:
        return results

    spy_returns = spy_returns_df.copy()
    spy_returns["date"] = pd.to_datetime(spy_returns["date"])
    results = results.merge(spy_returns, on="date", how="left")
    results = results[results["SPY_FutureReturn"].notna()].copy()
    if results.empty:
        return results

    results = results.rename(columns={label_column: "SelectedFutureReturn"})
    results["ExcessReturnVsSPY"] = results["SelectedFutureReturn"] - results["SPY_FutureReturn"]

    output_columns = [
        "date",
        "ticker",
        "TotalScore",
        "MomentumScore",
        "TrendScore",
        "MeanReversionScore",
        "VolatilityRiskScore",
        "SelectedFutureReturn",
        "SPY_FutureReturn",
        "ExcessReturnVsSPY",
        "FutureMaxReturn_30d",
        "FutureMaxDrawdown_30d",
        "Reached_5pct_Before_30d",
        "Reached_10pct_Before_30d",
        "Reached_15pct_Before_30d",
        "Hit_Negative_5pct_Before_30d",
        "Hit_Negative_10pct_Before_30d",
        "ReasonCodes",
    ]
    return results.loc[:, [column for column in output_columns if column in results.columns]].sort_values(
        ["date", "TotalScore"], ascending=[True, False]
    )


def summarize_backtest_results(results_df: pd.DataFrame) -> dict[str, float | int | str | None]:
    """Summarize selected-trade backtest results."""
    if results_df.empty:
        return {
            "Number of backtest dates": 0,
            "Number of selected trades": 0,
        }

    best_idx = results_df["SelectedFutureReturn"].idxmax()
    worst_idx = results_df["SelectedFutureReturn"].idxmin()
    return {
        "Number of backtest dates": int(results_df["date"].nunique()),
        "Number of selected trades": int(len(results_df)),
        "Average selected future return": float(results_df["SelectedFutureReturn"].mean()),
        "Median selected future return": float(results_df["SelectedFutureReturn"].median()),
        "Win rate": float((results_df["SelectedFutureReturn"] > 0).mean()),
        "Average SPY future return": float(results_df["SPY_FutureReturn"].mean()),
        "Median SPY future return": float(results_df["SPY_FutureReturn"].median()),
        "Average excess return vs SPY": float(results_df["ExcessReturnVsSPY"].mean()),
        "Median excess return vs SPY": float(results_df["ExcessReturnVsSPY"].median()),
        "Percent of trades beating SPY": float((results_df["ExcessReturnVsSPY"] > 0).mean()),
        "Average FutureMaxReturn_30d": _safe_mean(results_df, "FutureMaxReturn_30d"),
        "Average FutureMaxDrawdown_30d": _safe_mean(results_df, "FutureMaxDrawdown_30d"),
        "Probability of reaching +5% within 30 days": _safe_mean(results_df, "Reached_5pct_Before_30d"),
        "Probability of reaching +10% within 30 days": _safe_mean(results_df, "Reached_10pct_Before_30d"),
        "Probability of hitting -5% within 30 days": _safe_mean(results_df, "Hit_Negative_5pct_Before_30d"),
        "Best selected trade": f"{results_df.loc[best_idx, 'ticker']} on {pd.to_datetime(results_df.loc[best_idx, 'date']).date()}",
        "Worst selected trade": f"{results_df.loc[worst_idx, 'ticker']} on {pd.to_datetime(results_df.loc[worst_idx, 'date']).date()}",
    }


def _safe_mean(df: pd.DataFrame, column: str) -> float | None:
    if column not in df.columns or df[column].dropna().empty:
        return None
    value = df[column].mean()
    return None if pd.isna(value) else float(value)


def summary_to_frame(summary: dict[str, float | int | str | None]) -> pd.DataFrame:
    """Convert summary metrics to a display-friendly table."""
    rows = [{"metric": key, "value": value} for key, value in summary.items()]
    return pd.DataFrame(rows)
