from __future__ import annotations

import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

from gorpiq.config import DB_PATH, DEFAULT_DATA_SOURCE, PRICE_COLUMNS


def connect(db_path: Path = DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database(db_path: Path = DB_PATH) -> None:
    with connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS prices_daily (
                date TEXT NOT NULL,
                ticker TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                adj_close REAL,
                volume INTEGER,
                data_source TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (date, ticker, data_source)
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_prices_daily_ticker_date
            ON prices_daily (ticker, date)
            """
        )


def upsert_prices(prices: pd.DataFrame, db_path: Path = DB_PATH) -> int:
    if prices.empty:
        return 0

    missing_columns = sorted(set(PRICE_COLUMNS) - set(prices.columns))
    if missing_columns:
        raise ValueError(f"Price data is missing required columns: {missing_columns}")

    prepared = prices.loc[:, PRICE_COLUMNS].copy()
    prepared["date"] = pd.to_datetime(prepared["date"]).dt.date.astype(str)
    prepared["ticker"] = prepared["ticker"].astype(str).str.upper().str.strip()
    prepared["data_source"] = prepared["data_source"].fillna(DEFAULT_DATA_SOURCE)
    prepared["volume"] = prepared["volume"].fillna(0).astype("int64")
    prepared["created_at"] = datetime.utcnow().isoformat(timespec="seconds")

    records = prepared.to_dict("records")
    with connect(db_path) as connection:
        connection.executemany(
            """
            INSERT INTO prices_daily (
                date, ticker, open, high, low, close, adj_close, volume, data_source, created_at
            )
            VALUES (
                :date, :ticker, :open, :high, :low, :close, :adj_close, :volume, :data_source, :created_at
            )
            ON CONFLICT(date, ticker, data_source) DO UPDATE SET
                open = excluded.open,
                high = excluded.high,
                low = excluded.low,
                close = excluded.close,
                adj_close = excluded.adj_close,
                volume = excluded.volume,
                created_at = excluded.created_at
            """,
            records,
        )
    return len(records)


def get_price_summary(db_path: Path = DB_PATH) -> pd.DataFrame:
    initialize_database(db_path)
    with connect(db_path) as connection:
        return pd.read_sql_query(
            """
            SELECT
                ticker,
                MIN(date) AS first_date,
                MAX(date) AS last_date,
                COUNT(*) AS rows,
                data_source
            FROM prices_daily
            GROUP BY ticker, data_source
            ORDER BY ticker
            """,
            connection,
        )


def load_prices(
    tickers: Iterable[str] | None = None,
    start_date: date | str | None = None,
    end_date: date | str | None = None,
    db_path: Path = DB_PATH,
) -> pd.DataFrame:
    initialize_database(db_path)
    clauses: list[str] = []
    params: list[str] = []

    if tickers:
        normalized = [ticker.upper().strip() for ticker in tickers if ticker and ticker.strip()]
        if normalized:
            placeholders = ",".join("?" for _ in normalized)
            clauses.append(f"ticker IN ({placeholders})")
            params.extend(normalized)

    if start_date:
        clauses.append("date >= ?")
        params.append(str(start_date))

    if end_date:
        clauses.append("date <= ?")
        params.append(str(end_date))

    where_clause = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    query = f"""
        SELECT date, ticker, open, high, low, close, adj_close, volume, data_source, created_at
        FROM prices_daily
        {where_clause}
        ORDER BY ticker, date
    """

    with connect(db_path) as connection:
        return pd.read_sql_query(query, connection, params=params, parse_dates=["date"])
