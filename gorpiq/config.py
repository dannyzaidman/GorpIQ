from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXPORTS_DATA_DIR = DATA_DIR / "exports"
DB_PATH = DATA_DIR / "gorpiq.sqlite"

DEFAULT_DATA_SOURCE = "yfinance"

PRICE_COLUMNS = [
    "date",
    "ticker",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
    "data_source",
]

FEATURE_COLUMNS = [
    "date",
    "ticker",
    "Return_1d",
    "Return_2d",
    "Return_3d",
    "Return_5d",
    "Return_10d",
    "Return_15d",
    "Return_20d",
    "Return_30d",
    "Return_50d",
    "Return_100d",
    "Return_200d",
    "Return_YTD",
    "SMA_10",
    "SMA_20",
    "SMA_50",
    "SMA_100",
    "SMA_200",
    "Distance_SMA_10",
    "Distance_SMA_20",
    "Distance_SMA_50",
    "Distance_SMA_100",
    "Distance_SMA_200",
    "RSI_5",
    "RSI_14",
    "RSI_21",
    "ATR_14",
    "ATR_20",
    "ATR_Pct_14",
    "ATR_Pct_20",
]

LABEL_COLUMNS = [
    "date",
    "ticker",
    "FutureReturn_2d",
    "FutureReturn_3d",
    "FutureReturn_5d",
    "FutureReturn_10d",
    "FutureReturn_15d",
    "FutureReturn_20d",
    "FutureReturn_30d",
    "FutureMaxReturn_30d",
    "FutureMaxDrawdown_30d",
    "FutureDaysToPeak_30d",
    "FutureDaysToMaxDrawdown_30d",
    "Reached_5pct_Before_30d",
    "Reached_10pct_Before_30d",
    "Reached_15pct_Before_30d",
    "Hit_Negative_5pct_Before_30d",
    "Hit_Negative_10pct_Before_30d",
]

DISCLAIMER_TEXT = (
    "GorpIQ is for personal education and decision support only. "
    "It is not financial advice, does not guarantee results, and must not be "
    "used as an auto-trading system."
)
