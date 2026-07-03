from __future__ import annotations

import re
from collections.abc import Iterable

from gorpiq.config import DATA_DIR, EXPORTS_DATA_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR


def ensure_data_directories() -> None:
    for directory in (DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, EXPORTS_DATA_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def normalize_tickers(raw_tickers: str | Iterable[str]) -> list[str]:
    if isinstance(raw_tickers, str):
        candidates = re.split(r"[\s,;]+", raw_tickers)
    else:
        candidates = []
        for item in raw_tickers:
            candidates.extend(re.split(r"[\s,;]+", str(item)))

    tickers: list[str] = []
    for candidate in candidates:
        ticker = candidate.strip().upper()
        if ticker and ticker not in tickers:
            tickers.append(ticker)
    return tickers
