from __future__ import annotations

import re
from collections.abc import Iterable

from gorpiq.config import DATA_DIR, EXPORTS_DATA_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR


def ensure_data_directories() -> None:
    for directory in (DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, EXPORTS_DATA_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def _dedupe_normalized_tickers(candidates: Iterable[str]) -> list[str]:
    tickers: list[str] = []
    for candidate in candidates:
        ticker = candidate.strip().upper()
        if ticker and ticker not in tickers:
            tickers.append(ticker)
    return tickers


def parse_tickers_from_text(text: str) -> list[str]:
    """Parse ticker text separated by commas, spaces, tabs, or new lines."""
    if not text:
        return []

    candidates = re.split(r"[\s,;]+", text)
    return _dedupe_normalized_tickers(candidates)


def combine_ticker_sources(manual_tickers: str | Iterable[str], csv_tickers: Iterable[str]) -> list[str]:
    """Combine manual and CSV ticker sources, preserving first-seen order."""
    if isinstance(manual_tickers, str):
        manual = parse_tickers_from_text(manual_tickers)
    else:
        manual = normalize_tickers(manual_tickers)

    csv = normalize_tickers(csv_tickers)
    return _dedupe_normalized_tickers([*manual, *csv])


def normalize_tickers(raw_tickers: str | Iterable[str]) -> list[str]:
    """Normalize ticker input while preserving compatibility with older callers."""
    if isinstance(raw_tickers, str):
        return parse_tickers_from_text(raw_tickers)

    candidates = []
    for item in raw_tickers:
        candidates.extend(parse_tickers_from_text(str(item)))
    return _dedupe_normalized_tickers(candidates)
