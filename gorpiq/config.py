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

DISCLAIMER_TEXT = (
    "GorpIQ is for personal education and decision support only. "
    "It is not financial advice, does not guarantee results, and must not be "
    "used as an auto-trading system."
)
