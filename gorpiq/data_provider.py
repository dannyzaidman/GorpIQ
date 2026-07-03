from __future__ import annotations

from datetime import date, timedelta
from typing import Iterable, Protocol

import pandas as pd

from gorpiq.config import DEFAULT_DATA_SOURCE, PRICE_COLUMNS
from gorpiq.utils import normalize_tickers


class DailyPriceProvider(Protocol):
    def download_daily_prices(self, tickers: Iterable[str], start_date: date, end_date: date) -> pd.DataFrame:
        """Return normalized daily OHLCV rows for the requested ticker/date range."""


class YFinanceDataProvider:
    """Daily price provider backed by yfinance/Yahoo Finance."""

    data_source = DEFAULT_DATA_SOURCE

    def download_daily_prices(self, tickers: Iterable[str], start_date: date, end_date: date) -> pd.DataFrame:
        normalized_tickers = normalize_tickers(tickers)
        if not normalized_tickers:
            return pd.DataFrame(columns=PRICE_COLUMNS)

        import yfinance as yf

        # yfinance treats end as exclusive, so add one calendar day for the UI's inclusive end date.
        inclusive_end = end_date + timedelta(days=1)
        raw = yf.download(
            tickers=normalized_tickers,
            start=start_date,
            end=inclusive_end,
            auto_adjust=False,
            actions=False,
            group_by="ticker",
            progress=False,
            threads=True,
        )

        if raw.empty:
            return pd.DataFrame(columns=PRICE_COLUMNS)

        frames: list[pd.DataFrame] = []
        for ticker in normalized_tickers:
            ticker_frame = self._extract_ticker_frame(raw, ticker, len(normalized_tickers) == 1)
            if ticker_frame.empty:
                continue
            frames.append(ticker_frame)

        if not frames:
            return pd.DataFrame(columns=PRICE_COLUMNS)

        prices = pd.concat(frames, ignore_index=True)
        prices = prices.dropna(subset=["date", "ticker", "adj_close"])
        prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)
        return prices.loc[:, PRICE_COLUMNS]

    def _extract_ticker_frame(self, raw: pd.DataFrame, ticker: str, single_ticker: bool) -> pd.DataFrame:
        if isinstance(raw.columns, pd.MultiIndex):
            if ticker not in raw.columns.get_level_values(0):
                return pd.DataFrame(columns=PRICE_COLUMNS)
            frame = raw[ticker].copy()
        elif single_ticker:
            frame = raw.copy()
        else:
            return pd.DataFrame(columns=PRICE_COLUMNS)

        if frame.empty:
            return pd.DataFrame(columns=PRICE_COLUMNS)

        frame = frame.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adj_close",
                "Volume": "volume",
            }
        )
        if "adj_close" not in frame.columns and "close" in frame.columns:
            frame["adj_close"] = frame["close"]

        frame = frame.reset_index()
        date_column = "Date" if "Date" in frame.columns else frame.columns[0]
        frame = frame.rename(columns={date_column: "date"})
        frame["ticker"] = ticker
        frame["data_source"] = self.data_source

        for column in PRICE_COLUMNS:
            if column not in frame.columns:
                frame[column] = pd.NA

        return frame.loc[:, PRICE_COLUMNS]
